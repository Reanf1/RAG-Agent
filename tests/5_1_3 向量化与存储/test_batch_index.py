"""5.1.3 向量化与存储：TestBatchIndex。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import io
import tempfile
import unittest
from unittest.mock import patch
import pymupdf
from docx import Document as WordDocument
from src.data_loader import create_import_tasks
from src.retrieval.vector_store import VectorStore, batch_build_index
from tests.helpers import SmallEmbeddings


class TestBatchIndex(unittest.TestCase):
    """真实加载器/分块/Chroma 联调；二维向量用于核对是否重复编码。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.raw_dir = Path(self.directory.name) / "raw"
        self.embeddings = SmallEmbeddings()
        self.store = VectorStore(Path(self.directory.name) / "index", self.embeddings)

    def run_batch(self, tasks, **kwargs):
        """复用实际入库流程与真实临时数据库。"""
        return list(batch_build_index(tasks, self.raw_dir, vector_store=self.store, **kwargs))


    def test_all_formats_complete_index_and_keep_sources(self):
        """四类真实文件完成整个流程，只有写入索引后才标记成功。"""
        word = WordDocument()
        word.add_paragraph("Word 神经网络摘要")
        buffer = io.BytesIO()
        word.save(buffer)
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((72, 72), "PDF neural network")
            pdf_data = pdf.tobytes()
        tasks = create_import_tasks([("论文.pdf", pdf_data), ("论文.docx", buffer.getvalue()),
                                     ("论文.txt", "农业论文正文".encode()), ("论文.md", b"# AI paper")])
        progress = self.run_batch(tasks)
        self.assertEqual(progress[-1], {"completed": 4, "total": 4})
        self.assertTrue(all(task["status"] == "success" and task["indexed"] for task in tasks))
        self.assertEqual(self.store.count(), sum(task["chunk_count"] for task in tasks))
        for task in tasks:
            self.assertEqual(task["attempts"], 1)
            self.assertEqual(task["processed_chunks"], task["chunk_count"])
            self.assertEqual(task["added_chunks"], task["chunk_count"])
            for document in self.store.list_chunks(task["documents"][0].metadata["doc_id"]):
                self.assertEqual(document.metadata["source"], task["path"])
                self.assertEqual(document.metadata["source_file"], task["name"])

    def test_new_document_keeps_old_index_and_only_encodes_new(self):
        """新增一篇文献保留旧块，重新构造客户端不重算旧向量。"""
        first = create_import_tasks([("first.md", b"# Neural network")])
        self.run_batch(first)
        old_chunks = self.store.list_chunks()
        other_embeddings = SmallEmbeddings()
        reopened = VectorStore(Path(self.directory.name) / "index", other_embeddings)
        second = create_import_tasks([("second.txt", "农业论文".encode())])
        list(batch_build_index(second, self.raw_dir, vector_store=reopened))
        self.assertEqual(other_embeddings.document_calls, [["农业论文"]])
        self.assertEqual(reopened.count(), 2)
        self.assertEqual(reopened.list_chunks(old_chunks[0].metadata["doc_id"]), old_chunks)


    def test_index_error_keeps_loaded_file_and_retry_skips_loading(self):
        """一个索引错误不阻断下一文档，重试保留加载结果并跳过成功任务。"""
        tasks = create_import_tasks([("retry.txt", b"First"), ("good.txt", b"Second")])
        real_add = self.store.add_chunks

        def fail_first(chunks):
            if chunks[0].metadata["source_file"] == "retry.txt":
                raise OSError("索引暂时不可写")
            return real_add(chunks)

        with patch.object(self.store, "add_chunks", side_effect=fail_first):
            self.run_batch(tasks)
        self.assertEqual([task["status"] for task in tasks], ["failed", "success"])
        self.assertTrue(Path(tasks[0]["path"]).is_file())
        self.assertTrue(tasks[0]["documents"])
        with patch("src.data_loader.load_document", side_effect=AssertionError("不应重载")):
            progress = self.run_batch(tasks, retry_failed=True)
        self.assertEqual(progress[-1], {"completed": 1, "total": 1})
        self.assertEqual([task["attempts"] for task in tasks], [2, 1])
        self.assertEqual(tasks[0]["error"], "")
        self.assertTrue(all(task["indexed"] for task in tasks))
        self.assertEqual(self.store.count(), 2)


if __name__ == "__main__":
    unittest.main()
