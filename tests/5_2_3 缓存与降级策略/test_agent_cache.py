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

    def search(self, question="ViT使用什么输入？", doc_id=None, cache=None, **kwargs):
        tools = get_available_tools(cache=self.cache if cache is None else cache, session_id="test-session", **kwargs)
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

    def test_length_stop_is_incomplete_in_tool_and_log_and_never_cached(self):
        """服务length终止保留部分答案，但工具、日志和缓存均不得认定完成。"""
        from src.utils.logger import read_rag_requests
        response = {"model": "mock", "done": True, "done_reason": "length",
                    "prompt_eval_count": 80, "eval_count": 512,
                    "message": {"content": "部分内容。[参考文档1]"}}
        with patch("src.generation.rag_pipeline.urlopen", return_value=BytesIO(json.dumps(response).encode())):
            result = self.search()
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(result["answer"])
        self.assertFalse(self.cache.entries)
        records, _ = read_rag_requests()
        self.assertEqual(records[-1]["status"], "incomplete")
        self.assertEqual(records[-1]["done_reason"], "length")

    def test_low_relevance_is_not_cached(self):
        self.retriever.search.return_value = [(self.documents[0], .01)]
        self.assertEqual(self.search()["status"], "needs_confirmation")
        self.search()
        self.assertFalse(self.cache.entries)
        self.assertFalse(self.model_calls)

    def test_pending_and_cache_share_start_scope_but_recheck_at_finish(self):
        """同次请求只计算起止两次范围，结束仍能发现生成期间的语料变化。"""
        from src.generation.cache import cache_scope
        with patch("src.generation.cache.cache_scope", wraps=cache_scope) as scope:
            self.search(pending={})
            self.assertEqual(scope.call_count, 2)
        self.cache.clear()
        def changed_scope(store):
            value = cache_scope(store)
            return value if scope.call_count == 1 else "生成期间变更"
        with patch("src.generation.cache.cache_scope", side_effect=changed_scope) as scope:
            result = self.search(pending={})
        self.assertEqual(result["status"], "answered")
        self.assertFalse(self.cache.entries)

    def test_cache_write_failure_preserves_generated_answer(self):
        with patch.object(self.cache, "put", side_effect=RuntimeError("模拟缓存编码失败")):
            result = self.search()
        self.assertEqual(result["status"], "answered")
        self.assertIn("缓存保存失败", result["warnings"][-1])
        self.assertTrue(result["citations"])

    def test_confirmed_low_candidate_uses_reviewed_snapshot_without_second_retrieval(self):
        pending = {}
        self.retriever.search.return_value = [(self.documents[0], .01)]
        low = self.search(pending=pending, request_question="请根据论文回答输入形式")
        snapshot = pending[low["confirmation_id"]]
        self.assertFalse(self.model_calls)
        self.assertEqual(snapshot["question"], "请根据论文回答输入形式")
        result = self.search(confirmation=snapshot)
        self.assertEqual(self.retriever.search.call_count, 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertIn("已按你的确认", result["answer"])
        self.assertEqual(result["citations"][0]["text"], low["references"][0]["text"])
        self.assertEqual(result["retrieval"]["status"], "confirmed")
        self.assertFalse(self.cache.entries)

    def test_changed_corpus_or_session_rejects_confirmation_before_generation(self):
        pending = {}
        self.retriever.search.return_value = [(self.documents[0], .01)]
        low = self.search(pending=pending)
        snapshot = pending[low["confirmation_id"]]
        wrong = {**snapshot, "session_id": "another-session"}
        with self.assertRaisesRegex(ValueError, "会话"):
            self.search(confirmation=wrong)
        self.store.delete_document(self.documents[1].metadata["doc_id"])
        with self.assertRaisesRegex(ValueError, "失效"):
            self.search(confirmation=snapshot)
        self.assertFalse(self.model_calls)

    def test_confirmation_session_entry_fixes_action_args_and_preserves_memory(self):
        from src.agent.memory import MemoryManager, run_session
        memory = MemoryManager(Path(self.directory.name) / "sessions.sqlite3")
        session = memory.create_session("alice")
        memory.append_turn("alice", session, "之前的问题", "之前的回答")
        pending = {}
        tools = get_available_tools(session_id=session, pending=pending)
        self.retriever.search.return_value = [(self.documents[0], .01)]
        low = tools[0].invoke({"question": "ViT输入？"})
        snapshot = pending[low["confirmation_id"]]
        tools = get_available_tools(session_id=session, confirmation=snapshot)
        with patch("src.agent.react_loop.urlopen") as http:
            events = list(run_session("ViT输入？", "alice", session, tools=tools, memory=memory,
                                      confirmed_rag_args={"question": "ViT输入？", "doc_id": None}))
        self.assertEqual(events[0]["route"], "confirmation")
        self.assertEqual(next(e for e in events if e["type"] == "tool_call")["args"], {"question": "ViT输入？", "doc_id": None})
        http.assert_not_called()  # 确认参数固定，完整RAG报告直接保留，无额外规划或改写。
        self.assertEqual(self.retriever.search.call_count, 1)
        self.assertIn("已按你的确认", events[-1]["full_response"])
        self.assertEqual(len(memory.get_messages("alice", session)), 4)
        with self.assertRaises(PermissionError):
            list(run_session("ViT输入？", "bob", session, tools=tools, memory=memory,
                             confirmed_rag_args={"question": "ViT输入？", "doc_id": None}))

    def test_waiting_observation_does_not_ask_model_to_approve_candidate(self):
        from src.agent.react_loop import _observe_events
        pending = {}
        self.retriever.search.return_value = [(self.documents[0], .01)]
        low = self.search(pending=pending)
        context = {"observations": [{"name": "knowledge_base_search", "status": "success", "result": low}]}
        with patch("src.agent.react_loop.urlopen") as http:
            events = list(_observe_events("ViT输入？", get_available_tools(), context, stream=True))
        self.assertFalse(events[-1]["task_complete"])
        self.assertEqual(events[-1]["decision"], "finish")
        http.assert_not_called()

    def test_confirmation_does_not_approve_a_different_followup_query(self):
        pending = {}
        self.retriever.search.return_value = [(self.documents[0], .01)]
        low = self.search(pending=pending)
        result = self.search("ViT数据集？", confirmation=pending[low["confirmation_id"]], pending=pending)
        self.assertEqual(result["status"], "needs_confirmation")
        self.assertFalse(self.model_calls)


if __name__ == "__main__":
    unittest.main()
