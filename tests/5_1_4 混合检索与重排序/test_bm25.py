"""5.1.4 混合检索与重排序：TestBM25Retriever。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import math
import tempfile
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever, tokenize
from tests.helpers import SmallEmbeddings


class TestBM25Retriever(unittest.TestCase):
    """真实 BM25 与临时 Chroma；数值按公式独立核对，不调用大模型。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.embeddings = SmallEmbeddings()
        self.store = VectorStore(self.directory.name, self.embeddings)
        self.chunks = [
            Document(page_content="BatchNormalization BatchNormalization", metadata={
                "chunk_id": "a1", "doc_id": "a", "source_file": "论文A.pdf", "page_number": 2}),
            Document(page_content="batchnormalization training training training training training", metadata={
                "chunk_id": "a2", "doc_id": "a", "source_file": "论文A.pdf", "page_number": 3}),
            Document(page_content="Adam optimizer", metadata={
                "chunk_id": "b1", "doc_id": "b", "source_file": "论文B.docx", "paragraph_index": 1}),
            Document(page_content="注意力机制", metadata={
                "chunk_id": "c1", "doc_id": "c", "source_file": "论文C.txt", "line_start": 1, "line_end": 1}),
            Document(page_content="卷积网络", metadata={
                "chunk_id": "d1", "doc_id": "d", "source_file": "论文D.md", "line_start": 1, "line_end": 1}),
        ]
        self.store.add_chunks(self.chunks)
        self.retriever = BM25Retriever(self.store)


    def test_tokenize_chinese_english_and_numbers(self):
        """沿用上游分词，保留完整英文术语，忽略标点，文档/问题统一小写。"""
        self.assertEqual(tokenize("BatchNormalization 注意力，BERT-base 2024!"),
                         ["batchnormalization", "bert", "base", "2024", "注", "意", "力"])


    def test_scores_follow_term_frequency_and_length_normalization(self):
        """用已知词频、文档频次和长度独立计算 BM25 分数，核验排序。"""
        found = self.retriever.search("BATCHNORMALIZATION", k=10)
        self.assertEqual([doc.metadata["chunk_id"] for doc, _ in found], ["a1", "a2"])
        average_length = (2 + 6 + 2 + 5 + 4) / 5
        idf = math.log((5 - 2 + 0.5) / (2 + 0.5))
        for (document, score), frequency, length in zip(found, (2, 1), (2, 6)):
            expected = idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / average_length))
            self.assertAlmostEqual(score, expected)
            self.assertEqual(document, self.chunks[0 if frequency == 2 else 1])


    def test_rebuild_after_add_delete_and_reopen(self):
        """新增/删除后从当前正文重建，重开自动恢复，不重新编码文档。"""
        new = Document(page_content="Reranker", metadata={"chunk_id": "new", "doc_id": "new"})
        self.store.add_chunks([new])
        self.assertEqual(self.retriever.rebuild(), 6)
        self.assertEqual(self.retriever.search("Reranker")[0][0], new)
        self.store.delete_document("a")
        self.assertEqual(self.retriever.rebuild(), 4)
        self.assertEqual(self.retriever.search("BatchNormalization"), [])
        with patch("src.retrieval.vector_store.get_embeddings", side_effect=AssertionError("BM25 不应加载模型")):
            reopened = BM25Retriever(VectorStore(self.directory.name))
            self.assertEqual(reopened.search("Reranker"), self.retriever.search("Reranker"))
        self.assertEqual(self.embeddings.query_calls, [])
        self.assertEqual(len(self.embeddings.document_calls), 2)


if __name__ == "__main__":
    unittest.main()
