"""5.1.1 文档加载与批量导入：TestBatchImport。"""

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
from src.data_loader import batch_import, create_import_tasks, load_document


class TestBatchImport(unittest.TestCase):
    """验证批量任务的独立状态、真实保存和有界重试。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.raw_dir = Path(self.directory.name) / "raw"

    def test_all_formats_saved_with_correct_sources(self):
        """四类文档均实际解析和保存，来源指向持久文件而非临时文件。"""
        word = WordDocument()
        word.add_paragraph("Word 摘要")
        buffer = io.BytesIO()
        word.save(buffer)
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((72, 72), "PDF abstract")
            pdf_data = pdf.tobytes()
        files = [("paper.pdf", pdf_data), ("paper.docx", buffer.getvalue()),
                 ("说明.txt", "文本摘要".encode("utf-8")), ("说明.md", b"# Abstract")]
        tasks = create_import_tasks(files)
        progress = list(batch_import(tasks, self.raw_dir))
        self.assertEqual(progress[-1], {"completed": 4, "total": 4})
        self.assertEqual([t["status"] for t in tasks], ["success"] * 4)
        for task in tasks:
            path = Path(task["path"])
            self.assertEqual(path.read_bytes(), task["data"])
            self.assertEqual(task["attempts"], 1)
            for document in task["documents"]:
                self.assertEqual(document.metadata["source"], str(path))
                self.assertEqual(document.metadata["source_file"], task["name"])

    def test_failure_does_not_stop_later_files_or_leave_failed_file(self):
        """中间文件损坏也继续导入后续文件，完成进度包含失败项。"""
        tasks = create_import_tasks([("first.txt", b"First"), ("bad.pdf", b"broken"),
                                     ("last.md", b"# Last")])
        events = batch_import(tasks, self.raw_dir)
        next(events)
        self.assertEqual(tasks[0]["status"], "pending")
        next(events)
        self.assertEqual(tasks[0]["status"], "loading")
        progress = list(events)
        self.assertEqual([t["status"] for t in tasks], ["success", "failed", "success"])
        self.assertIn("FileDataError", tasks[1]["error"])
        self.assertEqual(tasks[1]["documents"], [])
        self.assertEqual(tasks[1]["path"], "")
        self.assertEqual(progress[-1], {"completed": 3, "total": 3})
        self.assertEqual(len(list(self.raw_dir.rglob("*.*"))), 2)

    def test_retry_failed_recovers_without_reloading_success(self):
        """模拟一次临时错误，手动重试仅失败项，成功项尝试次数不增加。"""
        tasks = create_import_tasks([("good.md", b"Good"), ("retry.txt", b"Retry")])

        def fail_once(path):
            if path.name == "retry.txt":
                raise OSError("临时读取失败")
            return load_document(path)

        with patch("src.data_loader.load_document", side_effect=fail_once):
            list(batch_import(tasks, self.raw_dir))
        self.assertEqual([t["status"] for t in tasks], ["success", "failed"])
        progress = list(batch_import(tasks, self.raw_dir, retry_failed=True))
        self.assertEqual(progress[-1], {"completed": 1, "total": 1})
        self.assertEqual([t["status"] for t in tasks], ["success", "success"])
        self.assertEqual([t["attempts"] for t in tasks], [1, 2])
        self.assertEqual(tasks[1]["error"], "")

    def test_persistent_failure_is_bounded(self):
        """每次重试只处理一轮，永久错误不会产生无限循环。"""
        tasks = create_import_tasks([("bad.txt", b"\xff")])
        list(batch_import(tasks, self.raw_dir))
        list(batch_import(tasks, self.raw_dir, retry_failed=True))
        self.assertEqual(tasks[0]["attempts"], 2)
        self.assertEqual(tasks[0]["status"], "failed")
        self.assertIn("UnicodeDecodeError", tasks[0]["error"])

    def test_repeated_run_skips_completed_tasks(self):
        """成功后再次调用不会重复加载或重复保存。"""
        tasks = create_import_tasks([("ok.txt", b"Ready")])
        list(batch_import(tasks, self.raw_dir))
        with patch("src.data_loader.load_document") as loader:
            self.assertEqual(list(batch_import(tasks, self.raw_dir)), [{"completed": 0, "total": 0}])
            list(batch_import(tasks, self.raw_dir, retry_failed=True))
            loader.assert_not_called()
        self.assertEqual(tasks[0]["attempts"], 1)

    def test_interrupted_loading_task_can_resume(self):
        """页面中断后遗留的处理中任务可继续处理。"""
        tasks = create_import_tasks([("ok.txt", b"Ready")])
        events = batch_import(tasks, self.raw_dir)
        next(events)
        next(events)
        events.close()
        self.assertEqual(tasks[0]["status"], "loading")
        list(batch_import(tasks, self.raw_dir))
        self.assertEqual(tasks[0]["status"], "success")

    def test_empty_batch_and_duplicate_inputs(self):
        """空批次不创建目录，相同文件名和内容的重复项合并。"""
        self.assertEqual(list(batch_import([], self.raw_dir)), [{"completed": 0, "total": 0}])
        self.assertFalse(self.raw_dir.exists())
        tasks = create_import_tasks([("same.txt", b"Same"), ("same.txt", b"Same")])
        self.assertEqual(len(tasks), 1)

    def test_same_name_different_content_is_preserved(self):
        """不同内容的同名文献独立保存，不覆盖旧内容。"""
        tasks = create_import_tasks([("same.txt", b"First"), ("same.txt", b"Second")])
        list(batch_import(tasks, self.raw_dir))
        self.assertNotEqual(tasks[0]["path"], tasks[1]["path"])
        self.assertEqual([Path(t["path"]).read_bytes() for t in tasks], [b"First", b"Second"])

    def test_unsupported_format_and_unsafe_names(self):
        """路径名和不支持格式记录失败，不写到导入目录之外。"""
        names = ["../escape.txt", "/tmp/escape.txt", "folder\\escape.txt", "paper.exe"]
        tasks = create_import_tasks([(name, b"Text") for name in names])
        list(batch_import(tasks, self.raw_dir))
        self.assertTrue(all(t["status"] == "failed" for t in tasks))
        self.assertFalse(self.raw_dir.exists())

    def test_size_limit(self):
        """前端之外的调用也核验单份文档大小。"""
        tasks = create_import_tasks([("large.txt", b"A" * (1024 * 1024 + 1))])
        list(batch_import(tasks, self.raw_dir, max_file_size_mb=1))
        self.assertEqual(tasks[0]["status"], "failed")
        self.assertIn("文件过大", tasks[0]["error"])

    def test_existing_file_with_different_bytes_is_not_overwritten(self):
        """已有保存位置被外部修改时明确失败，保留其内容。"""
        tasks = create_import_tasks([("paper.txt", b"Original")])
        list(batch_import(tasks, self.raw_dir))
        path = Path(tasks[0]["path"])
        path.write_bytes(b"Changed externally")
        tasks = create_import_tasks([("paper.txt", b"Original")])
        list(batch_import(tasks, self.raw_dir))
        self.assertEqual(tasks[0]["status"], "failed")
        self.assertEqual(path.read_bytes(), b"Changed externally")

    def test_storage_failure_is_retryable(self):
        """保存位置暂时不可用时记录错误，恢复后可重试成功。"""
        self.raw_dir.write_bytes(b"not a directory")
        tasks = create_import_tasks([("ok.txt", b"Text")])
        list(batch_import(tasks, self.raw_dir))
        self.assertEqual(tasks[0]["status"], "failed")
        self.raw_dir.unlink()
        list(batch_import(tasks, self.raw_dir, retry_failed=True))
        self.assertEqual(tasks[0]["status"], "success")


if __name__ == "__main__":
    unittest.main()
