"""5.1.3 向量化与存储：TestVectorStore。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tempfile
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from src.data_loader import load_document
from src.chunking import split_documents
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

    def test_empty_store_and_empty_query(self):
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(self.store.search("神经网络"), [])
        self.store.add_chunks(self.chunks)
        self.assertEqual(self.store.search(" \n"), [])
        self.assertEqual(self.store.search("神经网络", doc_id="missing"), [])
        self.assertEqual(self.embeddings.query_calls, [])

    def test_text_and_source_roundtrip(self):
        self.assertEqual(self.store.add_chunks(self.chunks), 3)
        actual = {chunk.metadata["chunk_id"]: chunk for chunk in self.store.list_chunks()}
        self.assertEqual(actual, {chunk.metadata["chunk_id"]: chunk for chunk in self.chunks})
        self.assertEqual(len(self.store.list_chunks("a")), 2)
        self.assertEqual(self.store.list_chunks("missing"), [])

    def test_cosine_order_and_negative_score(self):
        self.store.add_chunks(self.chunks)
        found = self.store.search("神经网络", k=10)
        self.assertEqual([doc.metadata["chunk_id"] for doc, _ in found], ["a1", "b1", "a2"])
        for (_, score), expected in zip(found, (1, 0, -1)):
            self.assertAlmostEqual(score, expected)

    def test_filtered_search_uses_persisted_vectors_without_hnsw(self):
        """小文档过滤绕开HNSW图；使用已有向量，不能重算文档或返回其他文档。"""
        self.store.add_chunks(self.chunks)
        before = list(self.embeddings.document_calls)
        with patch.object(self.store._store, "similarity_search_with_score",
                          side_effect=RuntimeError("Cannot return the results in a contigious 2D array")):
            found = self.store.search("农业", k=5, doc_id="b")
        self.assertEqual([doc.metadata["chunk_id"] for doc, _ in found], ["b1"])
        self.assertAlmostEqual(found[0][1], 1.0)
        self.assertEqual(self.embeddings.document_calls, before)

    def test_document_filter_and_top_k(self):
        self.store.add_chunks(self.chunks)
        self.assertEqual(len(self.store.search("神经网络", k=1)), 1)
        found = self.store.search("神经网络", doc_id="b")
        self.assertEqual([doc.metadata["chunk_id"] for doc, _ in found], ["b1"])

    def test_missing_hnsw_label_recovers_only_selected_document_without_writing(self):
        """Windows复测：原文可读但向量Label缺失，只读临时恢复并保留来源与提示。"""
        self.store.add_chunks(self.chunks)
        original_get = self.store._store.get
        before = self.store.list_chunks()
        def broken_vectors(**kwargs):
            if "embeddings" in kwargs.get("include", []):
                raise RuntimeError("Label not found")
            return original_get(**kwargs)
        with patch.object(self.store._store, "get", side_effect=broken_vectors), \
                patch.object(self.store._store, "add_documents", side_effect=AssertionError("不可改写索引")), \
                patch.object(self.store._store._collection, "delete", side_effect=AssertionError("不可删除原索引")):
            with self.assertLogs("src.retrieval.vector_store", level="WARNING"):
                found = self.store.search("神经网络", k=5, doc_id="a")
            self.assertEqual(self.store.search("问题", doc_id="missing"), [])
        self.assertEqual([doc.metadata["chunk_id"] for doc, _ in found], ["a1", "a2"])
        self.assertEqual([score for _, score in found], [1.0, -1.0])
        self.assertEqual(self.embeddings.document_calls[-1], ["神经网络论文", "反向向量论文"])
        self.assertIn("未修改原索引", found[0][0].metadata["retrieval_warning"])
        self.assertEqual(found[0][0].metadata["page_number"], 2)
        self.assertEqual(self.store.count(), 3)
        self.assertEqual(self.store.list_chunks(), before)

    def test_unrelated_vector_error_is_not_hidden_by_recovery(self):
        """只匹配已复现的Label异常，其他读取故障必须保留。"""
        self.store.add_chunks(self.chunks)
        with patch.object(self.store._store, "get", side_effect=RuntimeError("数据库读取失败")), \
                patch.object(self.embeddings, "embed_documents") as encode:
            with self.assertRaisesRegex(RuntimeError, "数据库读取失败"):
                self.store.search("神经网络", doc_id="a")
            encode.assert_not_called()

    def test_recovery_model_failure_is_not_an_empty_result(self):
        """恢复依赖同版本本地模型，编码失败仍上报，原文和块数保留。"""
        self.store.add_chunks(self.chunks)
        original_get = self.store._store.get
        def broken_vectors(**kwargs):
            if "embeddings" in kwargs.get("include", []):
                raise RuntimeError("Label not found")
            return original_get(**kwargs)
        with patch.object(self.store._store, "get", side_effect=broken_vectors), \
                patch.object(self.embeddings, "embed_documents", side_effect=RuntimeError("本地编码失败")):
            with self.assertRaisesRegex(RuntimeError, "本地编码失败"):
                self.store.search("神经网络", doc_id="a")
        self.assertEqual(self.store.count(), 3)
        self.assertEqual(len(self.store.list_chunks("a")), 2)

    def test_default_top_k_and_query_change_without_reembedding_documents(self):
        """默认 K 来自配置；新查询只编码问题，沿用持久化文档向量。"""
        from src.utils.config import load_config
        config = load_config()
        config["retrieval"]["top_k"] = 2
        self.store.add_chunks(self.chunks)
        with patch("src.retrieval.vector_store.load_config", return_value=config):
            reopened = VectorStore(self.directory.name, self.embeddings)
        self.assertEqual([doc.metadata["chunk_id"] for doc, _ in reopened.search("神经网络")],
                         ["a1", "b1"])
        self.assertEqual(reopened.search("农业", k=1)[0][0], self.chunks[1])
        self.assertEqual(self.embeddings.query_calls, ["神经网络", "农业"])
        self.assertEqual(self.embeddings.document_calls, [[chunk.page_content for chunk in self.chunks]])

    def test_blank_query_does_not_read_database(self):
        """空问题直接结束，不依赖数据库连接或查询编码。"""
        with patch.object(self.store, "count", side_effect=RuntimeError("数据库不可用")):
            self.assertEqual(self.store.search(" \n\t"), [])
        self.assertEqual(self.embeddings.query_calls, [])

    def test_readonly_operations_do_not_load_model(self):
        """只读正文、重复块和无有效向量查询无需权重，真正查询才加载。"""
        self.store.add_chunks(self.chunks)
        with patch("src.retrieval.vector_store.get_embeddings", side_effect=FileNotFoundError("模型未准备")) as model:
            readonly = VectorStore(self.directory.name)
            self.assertEqual(readonly.count(), 3)
            self.assertEqual(len(readonly.list_chunks()), 3)
            self.assertEqual(readonly.add_chunks(self.chunks), 0)
            self.assertEqual(readonly.delete_document("missing"), 0)
            self.assertEqual(readonly.search("  "), [])
            self.assertEqual(readonly.search("问题", doc_id="missing"), [])
            model.assert_not_called()
            with self.assertRaisesRegex(FileNotFoundError, "模型未准备"):
                readonly.search("神经网络")
            model.assert_called_once()

    def test_duplicate_import_does_not_encode_again(self):
        self.assertEqual(self.store.add_chunks(self.chunks + self.chunks), 3)
        self.assertEqual(self.store.add_chunks(self.chunks), 0)
        self.assertEqual(self.store.add_chunks([]), 0)
        self.assertEqual(self.embeddings.document_calls, [[chunk.page_content for chunk in self.chunks]])
        self.assertEqual(self.store.count(), 3)

    def test_incremental_import_only_encodes_new_chunks(self):
        self.store.add_chunks(self.chunks[:1])
        self.assertEqual(self.store.add_chunks(self.chunks), 2)
        self.assertEqual(self.embeddings.document_calls, [["神经网络论文"], ["农业数据论文", "反向向量论文"]])
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

    def test_chroma_receives_alias_without_resolving_it_back(self):
        """实际客户端保留英文入口；对外目录仍为原索引，便于状态旁注核对。"""
        self.store.add_chunks(self.chunks)
        alias = Path(self.directory.name) / "alias"
        alias.symlink_to(self.directory.name, target_is_directory=True)
        with patch("src.retrieval.vector_store.chroma_persist_path", return_value=str(alias)) as path:
            reopened = VectorStore(self.directory.name, SmallEmbeddings())
        path.assert_called_once_with(Path(self.directory.name).resolve())
        self.assertEqual(reopened._store._persist_directory, str(alias))
        self.assertEqual(reopened.directory, self.store.directory)
        self.assertEqual(reopened.search("神经网络", k=1)[0][0], self.chunks[0])

    def test_changed_model_or_index_parameters_are_rejected(self):
        from src.utils.config import load_config

        for section, key, value in (("embedding", "revision", "different-version"),
                                    ("retrieval", "search_ef", 10)):
            config = load_config()
            config[section][key] = value
            with self.subTest(key=key), patch("src.retrieval.vector_store.load_config", return_value=config):
                with self.assertRaisesRegex(ValueError, "使用新索引目录重建"):
                    VectorStore(self.directory.name, SmallEmbeddings())

    def test_invalid_ids_are_rejected_before_any_write(self):
        for metadata in ({}, {"chunk_id": "x"}, {"chunk_id": "", "doc_id": "a"}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.store.add_chunks([self.chunks[0], Document(page_content="坏块", metadata=metadata)])
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(self.embeddings.document_calls, [])
        for value in (None, "", "  "):
            with self.assertRaises(ValueError):
                self.store.delete_document(value)
        for k in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                self.store.search("问题", k=k)

    def test_same_text_in_different_documents_keeps_both_sources(self):
        second = Document(page_content=self.chunks[0].page_content,
                          metadata={"chunk_id": "c1", "doc_id": "c", "source_file": "论文C.pdf"})
        self.assertEqual(self.store.add_chunks([self.chunks[0], second]), 2)
        self.assertEqual({doc.metadata["doc_id"] for doc, _ in self.store.search("神经网络")}, {"a", "c"})

    def test_real_text_loading_and_batches_over_500(self):
        path = Path(self.directory.name) / "真实样本.md"
        path.write_text("# 神经网络\n\n论文使用农业数据。", encoding="utf-8")
        chunks = split_documents(load_document(path))
        self.assertEqual(self.store.add_chunks(chunks), len(chunks))
        self.assertEqual(self.store.search("农业")[0][0].metadata["source_file"], path.name)
        extra = [Document(page_content="神经网络", metadata={"chunk_id": f"extra-{i}", "doc_id": "extra"})
                 for i in range(501)]
        self.assertEqual(self.store.add_chunks(extra), 501)
        self.assertEqual(self.store.add_chunks(extra), 0)
        self.assertEqual(self.store.count(), 501 + len(chunks))


if __name__ == "__main__":
    unittest.main()
