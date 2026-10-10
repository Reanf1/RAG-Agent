"""5.1.3 向量化与存储：TestVectorStore。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tempfile
import unittest
from langchain_core.documents import Document
from src.retrieval.vector_store import VectorStore
from tests.helpers import SmallEmbeddings


class TestVectorStore(unittest.TestCase):
    """验证选定库的实际写入、查重、余弦查询、持久化和删除。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.embeddings = SmallEmbeddings()
        self.store = VectorStore(self.directory.name, self.embeddings)
        self.chunks = [
            Document(page_content="神经网络论文", metadata={
                "chunk_id": "a1", "doc_id": "a", "source_file": "论文A.pdf",
                "page_number": 2, "start_index": 0, "end_index": 7,
                "formula_layout": '[{"text":"x²","bbox":[1,2,3,4]}]',
            }),
            Document(page_content="农业数据论文", metadata={
                "chunk_id": "b1", "doc_id": "b", "source_file": "论文B.pdf", "page_number": 1,
            }),
            Document(page_content="反向向量论文", metadata={
                "chunk_id": "a2", "doc_id": "a", "source_file": "论文A.pdf", "page_number": 3,
            }),
        ]


    def test_text_and_source_roundtrip(self):
        self.assertEqual(self.store.add_chunks(self.chunks), 3)
        actual = {chunk.metadata["chunk_id"]: chunk for chunk in self.store.list_chunks()}
        self.assertEqual(actual, {chunk.metadata["chunk_id"]: chunk for chunk in self.chunks})
        self.assertEqual(len(self.store.list_chunks("a")), 2)
        self.assertEqual(self.store.list_chunks("missing"), [])


    def test_document_filter_and_top_k(self):
        self.store.add_chunks(self.chunks)
        self.assertEqual(len(self.store.search("神经网络", k=1)), 1)
        found = self.store.search("神经网络", doc_id="b")
        self.assertEqual([doc.metadata["chunk_id"] for doc, _ in found], ["b1"])


    def test_duplicate_import_does_not_encode_again(self):
        self.assertEqual(self.store.add_chunks(self.chunks + self.chunks), 3)
        self.assertEqual(self.store.add_chunks(self.chunks), 0)
        self.assertEqual(self.store.add_chunks([]), 0)
        self.assertEqual(self.embeddings.document_calls, [[chunk.page_content for chunk in self.chunks]])
        self.assertEqual(self.store.count(), 3)


    def test_delete_document_and_repeat_delete(self):
        self.store.add_chunks(self.chunks)
        self.assertEqual(self.store.delete_document("a"), 2)
        self.assertEqual(self.store.delete_document("a"), 0)
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.store.list_chunks("a"), [])
        self.assertEqual([doc.metadata["doc_id"] for doc, _ in self.store.search("神经网络")], ["b"])

    def test_reopen_preserves_content_without_reembedding(self):
        self.store.add_chunks(self.chunks)
        other_embeddings = SmallEmbeddings()
        reopened = VectorStore(self.directory.name, other_embeddings)
        self.assertEqual(reopened.count(), 3)
        self.assertEqual(reopened.search("神经网络", k=1)[0][0], self.chunks[0])
        self.assertEqual(other_embeddings.document_calls, [])


if __name__ == "__main__":
    unittest.main()
