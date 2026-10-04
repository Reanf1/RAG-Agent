"""Agent核心RAG工具缓存回归：临时Chroma和原文，模拟生成与Embedding。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from io import BytesIO
import json
import tempfile
import unittest
from unittest.mock import patch

from src.agent.tools import get_available_tools
from src.generation.cache import SemanticCache
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config
from tests.helpers import SmallEmbeddings


class TestAgentCache(unittest.TestCase):
    def setUp(self):
        from src.data_loader import batch_import, create_import_tasks
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = load_config()
        for name in ("raw_documents", "vector_index", "logs"):
            self.config["paths"][name] = str(Path(self.directory.name) / name)
        for module in ("src.agent.tools", "src.generation.cache", "src.retrieval.vector_store",
                       "src.generation.rag_pipeline", "src.utils.logger"):
            patcher = patch(module + ".load_config", return_value=self.config)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.embedding = SmallEmbeddings()
        patcher = patch("src.generation.cache.get_embeddings", return_value=self.embedding)
        patcher.start()
        self.addCleanup(patcher.stop)
        tasks = create_import_tasks([("ViT.md", b"ViT uses patches."), ("DINO.md", b"DINO uses self supervision.")])
        list(batch_import(tasks, self.config["paths"]["raw_documents"]))
        self.documents = [task["documents"][0] for task in tasks]
        for i, doc in enumerate(self.documents):
            doc.metadata["chunk_id"] = f"cache-test-{i}"
        self.store = VectorStore(embeddings=self.embedding)
        self.store.add_chunks(self.documents)
        self.cache = SemanticCache()
        patcher = patch("src.retrieval.hybrid_retriever.HybridRetriever")
        self.retriever = patcher.start().return_value
        self.addCleanup(patcher.stop)
        self.retriever.search.return_value = [(self.documents[0], .9)]
        self.model_calls = []
        def response(*args, **kw):
            self.model_calls.append(args)
            return BytesIO(json.dumps({"model": "mock", "done": True, "done_reason": "stop",
                "prompt_eval_count": 80, "eval_count": 10, "message": {"content": "使用patch。[参考文档1]"}}).encode())
        patcher = patch("src.generation.rag_pipeline.urlopen", side_effect=response)
        patcher.start()
        self.addCleanup(patcher.stop)

    def search(self, question="ViT使用什么输入？", doc_id=None, cache=None):
        tools = get_available_tools(cache=self.cache if cache is None else cache, session_id="test-session")
        tool = next(item for item in tools if item.name == "knowledge_base_search")
        self.assertEqual(set(tool.args), {"question", "doc_id"})
        return tool.invoke({"question": question, "doc_id": doc_id})

    def test_exact_and_semantic_hits_skip_retrieval_generation_and_current_tokens(self):
        from src.utils.logger import read_rag_requests
        first = self.search()
        exact = self.search()
        semantic = self.search("ViT采用什么输入？")
        self.assertEqual(exact["cache"]["mode"], "exact")
        self.assertEqual(semantic["cache"]["mode"], "semantic")
        self.assertEqual(exact["citations"], first["citations"])
        self.assertEqual(exact["usage"]["eval_count"], 0)
        self.assertEqual(exact["original_usage"]["eval_count"], 10)
        self.assertEqual(exact["retrieval"]["status"], "cache")
        self.assertEqual(self.retriever.search.call_count, 1)
        self.assertEqual(len(self.model_calls), 1)
        records, invalid = read_rag_requests()
        self.assertEqual(invalid, 0)
        self.assertEqual(records[-1]["tokens"]["source"], "cache")
        self.assertEqual(records[-1]["session_id"], "test-session")

    def test_other_session_or_document_filter_never_reuses_answer(self):
        self.search(doc_id=self.documents[0].metadata["doc_id"])
        self.search(cache=SemanticCache(), doc_id=self.documents[0].metadata["doc_id"])
        self.retriever.search.return_value = [(self.documents[1], .9)]
        self.search(doc_id=self.documents[1].metadata["doc_id"])
        self.assertEqual(self.retriever.search.call_count, 3)

    def test_corpus_change_invalidates_and_contextual_question_bypasses_cache(self):
        self.search()
        self.store.delete_document(self.documents[1].metadata["doc_id"])
        self.search()
        self.search("刚才那篇论文使用什么输入？")
        self.search("刚才那篇论文使用什么输入？")
        self.assertEqual(self.retriever.search.call_count, 4)

    def test_low_relevance_is_not_cached(self):
        self.retriever.search.return_value = [(self.documents[0], .01)]
        self.assertEqual(self.search()["status"], "needs_confirmation")
        self.search()
        self.assertFalse(self.cache.entries)
        self.assertFalse(self.model_calls)

    def test_cache_write_failure_preserves_generated_answer(self):
        with patch.object(self.cache, "put", side_effect=RuntimeError("模拟缓存编码失败")):
            result = self.search()
        self.assertEqual(result["status"], "answered")
        self.assertIn("缓存保存失败", result["warnings"][-1])
        self.assertTrue(result["citations"])


if __name__ == "__main__":
    unittest.main()
