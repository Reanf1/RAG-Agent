"""5.1.3 向量化与存储：TestBatchIndex。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from copy import deepcopy
import io
import tempfile
import unittest
from unittest.mock import patch
import pymupdf
from docx import Document as WordDocument
from src.data_loader import batch_import, create_import_tasks, load_document
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

    def test_size_boundary_rejects_before_parse_and_keeps_next_valid_document(self):
        """真实PDF刚好1MiB可入库；超1字节先拒绝，重试不污染已有向量。"""
        limit = 1024 * 1024
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((72, 72), "Size boundary fixture")
            raw = pdf.tobytes()
        exact = raw + b" " * (limit - len(raw))
        tasks = create_import_tasks([("too_large.pdf", exact + b" "), ("accepted.pdf", exact)])
        with patch("src.data_loader.load_document", wraps=load_document) as loader:
            progress = self.run_batch(tasks, max_file_size_mb=1)
        self.assertEqual([call.args[0].name for call in loader.call_args_list], ["accepted.pdf"])
        self.assertEqual(progress[-1], {"completed": 2, "total": 2})
        self.assertEqual([task["status"] for task in tasks], ["failed", "success"])
        self.assertEqual(tasks[0]["path"], "")
        self.assertEqual(tasks[0]["documents"], [])
        self.assertFalse(tasks[0]["indexed"])
        self.assertEqual(Path(tasks[1]["path"]).stat().st_size, limit)
        before = self.store.list_chunks()
        calls = deepcopy(self.embeddings.document_calls)
        with patch("src.data_loader.load_document", side_effect=AssertionError("超限文件不应进入解析")):
            self.run_batch(tasks, max_file_size_mb=1, retry_failed=True)
        self.assertEqual(tasks[0]["attempts"], 2)
        self.assertIn("文件过大", tasks[0]["error"])
        self.assertEqual(self.store.list_chunks(), before)
        self.assertEqual(self.embeddings.document_calls, calls)

    def test_delete_waits_for_import_and_removes_its_last_write(self):
        """阻塞真实入库编码，再并发删除；删除完成后不能留下迟到块。"""
        from threading import Event, Thread
        from src.frontend.components.documents import delete_document
        entered, release, deleted = Event(), Event(), Event()
        tasks = create_import_tasks([("race.txt", b"Concurrent import")])
        real_encode = self.embeddings.embed_documents

        def blocked_encode(texts):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("测试未释放编码")
            return real_encode(texts)

        errors = []
        def remove():
            try:
                delete_document(self.raw_dir, Path(self.directory.name) / "index",
                                tasks[0]["documents"][0].metadata["doc_id"])
                deleted.set()
            except Exception as error:
                errors.append(error)

        with patch.object(self.embeddings, "embed_documents", side_effect=blocked_encode):
            worker = Thread(target=self.run_batch, args=(tasks,))
            worker.start()
            self.assertTrue(entered.wait(5))
            remover = Thread(target=remove)
            remover.start()
            try:
                self.assertFalse(deleted.wait(0.1), "删除应等待同文档导入结束")
            finally:
                release.set()
                worker.join(5)
                remover.join(5)
        self.assertFalse(worker.is_alive() or remover.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(deleted.is_set())
        self.assertEqual(self.store.count(), 0)
        self.assertFalse(Path(tasks[0]["path"]).exists())

    def test_deleted_loaded_task_cannot_resume_old_index(self):
        """已加载任务被删除后，旧任务重试不能利用内存正文复活索引。"""
        from src.frontend.components.documents import delete_document
        tasks = create_import_tasks([("old.txt", b"Old task")])
        list(batch_import(tasks, self.raw_dir))
        delete_document(self.raw_dir, Path(self.directory.name) / "index",
                        tasks[0]["documents"][0].metadata["doc_id"])
        self.run_batch(tasks)
        self.assertEqual(tasks[0]["status"], "failed")
        self.assertIn("原文", tasks[0]["error"])
        self.assertEqual(self.store.count(), 0)

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

    def test_repeated_and_recreated_tasks_do_not_encode_old_chunks(self):
        """页面重跑跳过完成任务，重新上传相同原文也不重新编码。"""
        files = [("paper.txt", b"Neural network")]
        tasks = create_import_tasks(files)
        self.run_batch(tasks)
        with patch("src.data_loader.load_document") as loader:
            self.assertEqual(self.run_batch(tasks), [{"completed": 0, "total": 0}])
            loader.assert_not_called()
        again = create_import_tasks(files)
        self.run_batch(again)
        self.assertTrue(again[0]["indexed"])
        self.assertEqual(again[0]["added_chunks"], 0)
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.embeddings.document_calls, [["Neural network"]])

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

    def test_partial_batch_failure_only_retries_missing_chunks(self):
        """第 501 块故障后保留前 500 块，下一次只编码剩余块。"""
        tasks = create_import_tasks([("large.txt", ("神经网络。\n\n" * 200).encode())])
        from src.chunking.fixed_chunk import split_fixed

        # 小窗口获得 501+ 个真实原文块，便于验证批次边界而不加载真实大模型。
        with patch("src.chunking.split_documents", side_effect=lambda docs: split_fixed(docs, 2, 0)):
            real_add = self.store.add_chunks
            calls = 0

            def fail_second(chunks):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("第二批失败")
                return real_add(chunks)

            with patch.object(self.store, "add_chunks", side_effect=fail_second):
                self.run_batch(tasks)
            self.assertEqual(tasks[0]["status"], "failed")
            self.assertFalse(tasks[0]["indexed"])
            self.assertEqual(tasks[0]["processed_chunks"], 500)
            self.assertEqual(self.store.count(), 500)
            self.run_batch(tasks, retry_failed=True)
        count = tasks[0]["chunk_count"]
        self.assertGreater(count, 500)
        self.assertEqual(tasks[0]["status"], "success")
        self.assertEqual(self.store.count(), count)
        self.assertEqual(tasks[0]["added_chunks"], count - 500)
        self.assertEqual(sum(len(call) for call in self.embeddings.document_calls), count)

    def test_stage_progress_and_interrupted_index_resume(self):
        """观察分块/索引阶段；进度中断后可恢复，不误标为成功。"""
        tasks = create_import_tasks([("paper.txt", b"Neural network")])
        events = batch_build_index(tasks, self.raw_dir, vector_store=self.store)
        statuses = []
        for _ in events:
            statuses.append(tasks[0]["status"])
            if tasks[0]["status"] == "indexing":
                break
        events.close()
        self.assertIn("loading", statuses)
        self.assertIn("chunking", statuses)
        self.assertFalse(tasks[0]["indexed"])
        self.run_batch(tasks)
        self.assertTrue(tasks[0]["indexed"])
        self.assertEqual(self.store.count(), 1)

    def test_loading_failures_and_empty_batch_do_not_initialize_model(self):
        """空批次/解码失败无需模型，也不创建索引；加载错误仍可有界重试。"""
        with patch("src.retrieval.vector_store.VectorStore") as factory:
            self.assertEqual(list(batch_build_index([], self.raw_dir)), [{"completed": 0, "total": 0}])
            tasks = create_import_tasks([("bad.txt", b"\xff")])
            list(batch_build_index(tasks, self.raw_dir))
            list(batch_build_index(tasks, self.raw_dir, retry_failed=True))
            factory.assert_not_called()
        self.assertEqual(tasks[0]["attempts"], 2)
        self.assertIn("UnicodeDecodeError", tasks[0]["error"])

    def test_loaded_documents_can_be_indexed_without_reload(self):
        """原始加载接口的成功任务也可继续进入索引阶段。"""
        tasks = create_import_tasks([("paper.txt", b"Neural network")])
        list(batch_import(tasks, self.raw_dir))
        with patch("src.data_loader.load_document", side_effect=AssertionError("不应重载")):
            self.run_batch(tasks)
        self.assertTrue(tasks[0]["indexed"])
        self.assertEqual(tasks[0]["attempts"], 2)

    def test_empty_content_after_loading_is_not_index_success(self):
        """加载后没有有效分块不能算索引成功；保留原文供用户核对。"""
        tasks = create_import_tasks([("paper.txt", b"Neural network")])
        with patch("src.chunking.split_documents", return_value=[]):
            self.run_batch(tasks)
        self.assertEqual(tasks[0]["status"], "failed")
        self.assertFalse(tasks[0]["indexed"])
        self.assertTrue(Path(tasks[0]["path"]).is_file())
        self.assertIn("没有可索引", tasks[0]["error"])
        self.assertEqual(self.embeddings.document_calls, [])


if __name__ == "__main__":
    unittest.main()
