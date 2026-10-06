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

    def test_chinese_academic_query_matches_english_position_terms(self):
        """英文论文的位置编码可以通过中英学术术语匹配，原文不修改。"""
        original = Document(page_content="Positional embeddings retain spatial order.", metadata={
            "chunk_id": "position", "doc_id": "position", "source_file": "位置.pdf", "page_number": 3})
        self.store.add_chunks([original])
        self.retriever.rebuild()
        found = self.retriever.search("位置编码如何保留顺序？")
        self.assertEqual(found[0][0], original)

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

    def test_chinese_and_source_preservation(self):
        """中文单字组合可检索，对应原文、位置和稳定 ID 全部保留。"""
        found = self.retriever.search("注意力")
        self.assertEqual([doc for doc, _ in found], [self.chunks[3]])
        self.assertGreater(found[0][1], 0)

    def test_fullwidth_pdf_terms_match_without_changing_original_text(self):
        """全角英文/数字常见于论文 PDF，词项归一化后可查且原文不变。"""
        fullwidth = Document(page_content="ＢＥＲＴ ２０２４", metadata={
            "chunk_id": "fullwidth", "doc_id": "fullwidth", "source_file": "全角论文.pdf", "page_number": 1})
        self.store.add_chunks([fullwidth])
        self.retriever.rebuild()
        for query in ("bert 2024", "ｂｅｒｔ ２０２４"):
            self.assertEqual(self.retriever.search(query)[0][0], fullwidth)

    def test_default_k_override_and_document_filter(self):
        """继承存储配置 K；文档过滤先于截断，分数仍基于完整语料。"""
        self.store.top_k = 1
        retriever = BM25Retriever(self.store)
        self.assertEqual(len(retriever.search("BatchNormalization")), 1)
        self.assertEqual(len(retriever.search("BatchNormalization", k=10)), 2)
        query = "Adam BatchNormalization"
        all_scores = {doc.metadata["chunk_id"]: score for doc, score in retriever.search(query, k=10)}
        found = retriever.search(query, doc_id="a")
        self.assertEqual(found[0][0].metadata["doc_id"], "a")
        self.assertEqual(found[0][1], all_scores[found[0][0].metadata["chunk_id"]])
        self.assertEqual(retriever.search(query, doc_id="missing"), [])

    def test_empty_and_unknown_queries_do_not_return_unrelated_documents(self):
        """无匹配时返回空列表，英文术语按词匹配而非子串匹配。"""
        for query in ("  \n", "！？…", "unknownkeyword", "Normalization"):
            with self.subTest(query=query):
                self.assertEqual(self.retriever.search(query), [])
        self.assertEqual(self.embeddings.query_calls, [])
        self.assertEqual(self.embeddings.document_calls, [[doc.page_content for doc in self.chunks]])

    def test_empty_and_punctuation_only_corpus(self):
        """空库和全无词语料不构造 BM25，混入标点块不影响有效块查询。"""
        store = VectorStore(Path(self.directory.name) / "empty", self.embeddings)
        retriever = BM25Retriever(store)
        self.assertEqual(retriever.search("Adam"), [])
        store.add_chunks([Document(page_content="！？…", metadata={"chunk_id": "p1", "doc_id": "p"})])
        self.assertEqual(retriever.rebuild(), 0)
        self.assertEqual(retriever.search("Adam"), [])
        store.add_chunks([self.chunks[2]])
        self.assertEqual(retriever.rebuild(), 1)
        self.assertEqual(retriever.search("Adam")[0][0], self.chunks[2])

    def test_zero_and_negative_scores_still_keep_keyword_matches(self):
        """复现上游 >0 过滤的漏检：单文档负分、两文档零分均应命中。"""
        for index, texts in enumerate((["BatchNormalization"], ["BatchNormalization", "Adam"])):
            with self.subTest(texts=texts):
                store = VectorStore(Path(self.directory.name) / f"small-{index}", self.embeddings)
                store.add_chunks([Document(page_content=text, metadata={"chunk_id": str(i), "doc_id": str(i)})
                                  for i, text in enumerate(texts)])
                found = BM25Retriever(store).search("BatchNormalization")
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0][0].page_content, "BatchNormalization")
                if index == 0:
                    self.assertLess(found[0][1], 0)
                else:
                    self.assertEqual(found[0][1], 0)

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

    def test_same_content_keeps_separate_document_sources(self):
        """不同来源的相同正文保留各自 ID，供后续按块融合使用。"""
        self.store.add_chunks([Document(page_content="Reranker", metadata={"chunk_id": f"r{i}", "doc_id": f"r{i}"})
                               for i in range(2)])
        self.retriever.rebuild()
        self.assertEqual({doc.metadata["doc_id"] for doc, _ in self.retriever.search("Reranker")}, {"r0", "r1"})

    def test_invalid_k_and_database_errors_are_not_hidden(self):
        """参数错误与正文读取失败直接上报，不静默回退向量查询。"""
        for k in (0, -1, 1.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                self.retriever.search("Adam", k=k)
        with patch.object(self.store, "list_chunks", side_effect=RuntimeError("数据库不可用")):
            with self.assertRaisesRegex(RuntimeError, "数据库不可用"):
                BM25Retriever(self.store)


if __name__ == "__main__":
    unittest.main()
