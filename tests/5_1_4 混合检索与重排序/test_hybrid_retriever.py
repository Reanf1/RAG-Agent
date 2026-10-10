"""5.1.4 混合检索与重排序：TestRRFFusion、TestHybridRetriever。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tempfile
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever, rrf_fusion
from tests.helpers import SmallEmbeddings


class TestRRFFusion(unittest.TestCase):
    """用明确排名和手算分数核验融合，不用原始分数冒充排名贡献。"""

    def setUp(self):
        self.documents = {key: Document(page_content=f"论文 {key}", metadata={
            "chunk_id": key, "doc_id": key, "source_file": f"{key}.pdf", "page_number": 1})
            for key in ("a", "b", "c", "d")}

    def test_one_based_rank_formula_and_shared_chunk_bonus(self):
        """两路共同命中的块分数累加，第一名贡献为 1/61。"""
        d = self.documents
        found = rrf_fusion([(d["a"], 0.9), (d["b"], 0.8), (d["c"], -0.2)],
                           [(d["b"], 8.0), (d["c"], 2.0), (d["d"], 0.0)])
        expected = {"a": 1 / 61, "b": 1 / 62 + 1 / 61,
                    "c": 1 / 63 + 1 / 62, "d": 1 / 63}
        self.assertEqual([doc.metadata["chunk_id"] for doc, _ in found], ["b", "c", "a", "d"])
        for document, score in found:
            self.assertAlmostEqual(score, expected[document.metadata["chunk_id"]])
            self.assertIs(document, d[document.metadata["chunk_id"]])


class TestHybridRetriever(unittest.TestCase):
    """真实 Chroma/BM25 联调，两维向量仅用于确定行为与模型调用次数。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.embeddings = SmallEmbeddings()
        self.store = VectorStore(self.directory.name, self.embeddings)
        self.chunks = [Document(page_content=text, metadata={
            "chunk_id": str(i), "doc_id": str(i), "source_file": f"论文{i}.pdf", "page_number": i + 1})
            for i, text in enumerate(("神经网络研究", "BatchNormalization BatchNormalization",
                                     "BatchNormalization training training training", "农业数据", "反向向量"))]
        self.store.add_chunks(self.chunks)
        self.retriever = HybridRetriever(self.store)

    def test_both_routes_take_candidates_before_final_top_k(self):
        """最终只取 1 个也先各召回 20 个候选，不提前把两路截为 Top-1。"""
        original_bm25_search = BM25Retriever.search
        with (patch.object(self.store, "search", wraps=self.store.search) as vector,
              patch.object(BM25Retriever, "search", autospec=True, side_effect=original_bm25_search) as bm25):
            found = self.retriever.search("BatchNormalization", k=1)
        vector.assert_called_once_with("BatchNormalization", k=20, doc_id=None)
        self.assertEqual(bm25.call_args.kwargs, {"k": 20, "doc_id": None})
        self.assertIs(bm25.call_args.args[0].vector_store, self.store)
        self.assertEqual(len(found), 1)
        self.assertIn(found[0][0], self.chunks[1:3])
        self.assertGreater(found[0][1], 1 / 61)
        self.assertEqual(self.embeddings.query_calls, ["BatchNormalization"])
        self.assertEqual(self.embeddings.document_calls, [[doc.page_content for doc in self.chunks]])


    def test_model_reranks_fused_candidates_before_final_top_k(self):
        """模型先看到融合 Top-20；候选末项也能经模型升到最终 Top-1。"""
        self.store.add_chunks([Document(page_content=f"论文候选 {i}", metadata={
            "chunk_id": f"extra-{i}", "doc_id": "extra"}) for i in range(25)])
        candidates = self.retriever.search("unknownkeyword", k=20)
        with patch("src.retrieval.reranker.get_reranker") as model:
            model.return_value.predict.return_value = [0.01] * 19 + [0.99]
            found = self.retriever.search("unknownkeyword", k=1, rerank=True)
            self.assertEqual(found, [(candidates[-1][0], 0.99)])
            model.return_value.predict.assert_called_once_with(
                [["unknownkeyword", doc.page_content] for doc, _ in candidates],
                batch_size=8, show_progress_bar=False)
            # K 大于默认候选数时相应扩大，而不是返回不足 K 的人为截断结果。
            model.return_value.predict.return_value = [0.5] * 25
            found = self.retriever.search("unknownkeyword", k=25, rerank=True)
            self.assertEqual(len(found), 25)
        self.assertEqual(len(self.embeddings.document_calls), 2)


if __name__ == "__main__":
    unittest.main()
