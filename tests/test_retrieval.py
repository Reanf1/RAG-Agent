"""模块一测试：使用临时文档验证文本与来源，不依赖模型或向量数据库。"""

import hashlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf
from docx import Document as WordDocument
from docx.opc.exceptions import PackageNotFoundError
from langchain_core.documents import Document

# 直接运行测试文件时，按文件位置加入项目根目录，不依赖当前工作目录。
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader.pdf_loader import load_pdf
from src.data_loader.docx_loader import load_docx
from src.data_loader.text_loader import load_text
from src.data_loader import batch_import, create_import_tasks, load_document


class TestPDFLoader(unittest.TestCase):
    """验证实际 PDF 解析、页码和常见导入失败，不用 mock 替代解析器。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "论文.PDF"

    def write_pdf(self, texts):
        """生成可重复的中文/英文测试页面，空字符串对应空白页。"""
        with pymupdf.open() as pdf:
            for text in texts:
                page = pdf.new_page()
                if text:
                    page.insert_text((72, 72), text, fontname="china-s")
            pdf.save(self.path)

    def test_chinese_text_and_page_metadata(self):
        """每页正文与来源完整，页码同时提供索引和展示值。"""
        self.write_pdf(["中文论文摘要", "实验结果与结论"])
        documents = load_pdf(self.path)
        self.assertEqual(len(documents), 2)
        self.assertEqual([d.page_content for d in documents], ["中文论文摘要", "实验结果与结论"])
        for index, document in enumerate(documents):
            self.assertIsInstance(document, Document)
            self.assertEqual(document.metadata, {
                "source": str(self.path.resolve()),
                "source_file": "论文.PDF",
                "file_type": ".pdf",
                "doc_id": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "page": index,
                "page_number": index + 1,
                "total_pages": 2,
            })

    def test_relative_string_path(self):
        """字符串相对路径也能加载，来源统一为绝对路径。"""
        self.write_pdf(["摘要"])
        documents = load_pdf(os.path.relpath(self.path))
        self.assertEqual(documents[0].metadata["source"], str(self.path.resolve()))

    def test_document_id_is_stable_and_shared(self):
        """重复导入与文件副本共用内容标识，各页不会随机生成不同 ID。"""
        self.write_pdf(["摘要", "结论"])
        copy_path = self.path.with_name("副本.pdf")
        copy_path.write_bytes(self.path.read_bytes())
        documents = load_pdf(self.path) + load_pdf(self.path) + load_pdf(copy_path)
        self.assertEqual(len({d.metadata["doc_id"] for d in documents}), 1)

    def test_document_id_changes_with_file_content(self):
        """文件内容改变后标识也改变，避免误用旧索引。"""
        self.write_pdf(["初始内容"])
        original_id = load_pdf(self.path)[0].metadata["doc_id"]
        self.path = self.path.with_name("更新.pdf")
        self.write_pdf(["更新内容"])
        self.assertNotEqual(load_pdf(self.path)[0].metadata["doc_id"], original_id)

    def test_position_sorting(self):
        """内容流先写下方再写上方时，正文仍按页面位置排序。"""
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 144), "BOTTOM")
            page.insert_text((72, 72), "TOP")
            pdf.save(self.path)
        text = load_pdf(self.path)[0].page_content
        self.assertLess(text.index("TOP"), text.index("BOTTOM"))

    def test_blank_page_does_not_shift_page_numbers(self):
        """跳过中间空白页仍保留原始物理页码，避免引用错页。"""
        self.write_pdf(["第一页", "", "第三页"])
        with self.assertLogs("src.data_loader.pdf_loader", level="WARNING") as logs:
            documents = load_pdf(self.path)
        self.assertEqual([d.metadata["page_number"] for d in documents], [1, 3])
        self.assertEqual([d.metadata["total_pages"] for d in documents], [3, 3])
        self.assertIn("第 2 页", logs.output[0])

    def test_image_only_pdf_is_not_silently_accepted(self):
        """将文字渲染为图片，验证没有文本层的扫描页不会伪装为导入成功。"""
        with pymupdf.open() as original:
            page = original.new_page()
            page.insert_text((72, 72), "Image-only paper")
            image = page.get_pixmap().tobytes("png")
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_image(page.rect, stream=image)
            pdf.save(self.path)
        with self.assertLogs("src.data_loader.pdf_loader", level="WARNING"):
            with self.assertRaisesRegex(ValueError, "未提取到文本"):
                load_pdf(self.path)

    def test_missing_file(self):
        """不存在的文件明确报错。"""
        with self.assertRaises(FileNotFoundError):
            load_pdf(self.path)

    def test_unsupported_extension(self):
        """其他加载器的格式不能误入 PDF 解析流程。"""
        path = self.path.with_suffix(".txt")
        path.write_text("摘要", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "不支持此文件格式"):
            load_pdf(path)

    def test_damaged_pdf(self):
        """损坏文件的解析错误向上传递，不返回空列表掩盖失败。"""
        self.path.write_bytes(b"not a valid PDF")
        with self.assertRaises(pymupdf.FileDataError):
            load_pdf(self.path)

    def test_password_protected_pdf(self):
        """需要打开密码的文件明确提示先解密。"""
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 72), "Protected paper")
            pdf.save(self.path, encryption=pymupdf.PDF_ENCRYPT_AES_256,
                     owner_pw="owner", user_pw="reader")
        with self.assertRaisesRegex(ValueError, "需要密码"):
            load_pdf(self.path)


class TestDOCXLoader(unittest.TestCase):
    """通过实际 DOCX 验证正文、表格顺序和位置，不修改原始课程资料。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "论文.DOCX"

    def test_paragraph_table_order_and_metadata(self):
        """表格保留在正文原位置，中文段落与单元格内容均可读取。"""
        word = WordDocument()
        word.add_paragraph("摘要")
        table = word.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "方法"
        table.cell(0, 1).text = "准确率"
        table.cell(1, 0).text = "Transformer"
        table.cell(1, 1).text = "90%"
        word.add_paragraph("结论")
        word.save(self.path)

        documents = load_docx(os.path.relpath(self.path))
        self.assertEqual([d.page_content for d in documents], [
            "摘要", "方法\t准确率\nTransformer\t90%", "结论",
        ])
        self.assertEqual([d.metadata["block_index"] for d in documents], [1, 2, 3])
        self.assertEqual([d.metadata["block_type"] for d in documents], [
            "paragraph", "table", "paragraph",
        ])
        self.assertEqual(documents[0].metadata["paragraph_index"], 1)
        self.assertEqual(documents[1].metadata["table_index"], 1)
        self.assertEqual(documents[2].metadata["paragraph_index"], 2)
        for document in documents:
            self.assertIsInstance(document, Document)
            self.assertEqual(document.metadata["source"], str(self.path.resolve()))
            self.assertEqual(document.metadata["source_file"], "论文.DOCX")
            self.assertEqual(document.metadata["file_type"], ".docx")
            self.assertNotIn("page", document.metadata)
            self.assertNotIn("page_number", document.metadata)

    def test_blank_blocks_do_not_shift_positions(self):
        """空段落和空表格跳过，但后续段落/表格编号不偏移。"""
        word = WordDocument()
        word.add_paragraph("")
        word.add_table(rows=1, cols=1)
        word.add_paragraph("正文")
        word.add_table(rows=1, cols=1).cell(0, 0).text = "实验数据"
        word.save(self.path)
        documents = load_docx(self.path)
        self.assertEqual([d.metadata["block_index"] for d in documents], [3, 4])
        self.assertEqual(documents[0].metadata["paragraph_index"], 2)
        self.assertEqual(documents[1].metadata["table_index"], 2)

    def test_table_empty_cells_and_multiple_paragraphs(self):
        """空单元格保留分隔符，单元格内的多个段落不丢失。"""
        word = WordDocument()
        table = word.add_table(rows=1, cols=3)
        table.cell(0, 1).text = "第一段"
        table.cell(0, 1).add_paragraph("第二段")
        word.save(self.path)
        self.assertEqual(load_docx(self.path)[0].page_content, "\t第一段\n第二段\t")

    def test_document_id_is_stable_shared_and_changes(self):
        """同文件各块共用内容 ID，重复读取稳定，内容变化后更新。"""
        word = WordDocument()
        word.add_paragraph("摘要")
        word.add_paragraph("结论")
        word.save(self.path)
        documents = load_docx(self.path) + load_docx(self.path)
        ids = {d.metadata["doc_id"] for d in documents}
        self.assertEqual(ids, {hashlib.sha256(self.path.read_bytes()).hexdigest()})
        word.add_paragraph("新增实验")
        word.save(self.path)
        self.assertNotIn(load_docx(self.path)[0].metadata["doc_id"], ids)

    def test_empty_document(self):
        """仅有空段落和空表格时明确报错。"""
        word = WordDocument()
        word.add_paragraph(" \t ")
        word.add_table(rows=1, cols=2)
        word.save(self.path)
        with self.assertRaisesRegex(ValueError, "未提取到文本"):
            load_docx(self.path)

    def test_missing_file(self):
        """缺失文件不会被当成空文档。"""
        with self.assertRaises(FileNotFoundError):
            load_docx(self.path)

    def test_legacy_doc_extension(self):
        """旧版 .doc 格式需转换后导入。"""
        path = self.path.with_suffix(".doc")
        path.write_bytes(b"legacy document")
        with self.assertRaisesRegex(ValueError, "仅支持 .docx"):
            load_docx(path)

    def test_damaged_document(self):
        """损坏文件的解析错误向上传递。"""
        self.path.write_bytes(b"not a DOCX archive")
        with self.assertRaises(PackageNotFoundError):
            load_docx(self.path)


class TestTextLoader(unittest.TestCase):
    """验证纯文本保真、中文编码和行范围。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "文献.txt"

    def test_utf8_text_and_line_metadata(self):
        """中文正文保留首尾空白与空行，来源和行号完整。"""
        text = "\n  中文正文\n\n第二段\n"
        self.path.write_bytes(text.encode("utf-8"))
        documents = load_text(os.path.relpath(self.path))
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, text)
        self.assertEqual(documents[0].metadata, {
            "source": str(self.path.resolve()),
            "source_file": "文献.txt",
            "file_type": ".txt",
            "doc_id": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "line_start": 1,
            "line_end": 4,
        })

    def test_utf8_bom_and_crlf(self):
        """移除 BOM，但保留 Windows 的 CRLF 换行。"""
        text = "第一行\r\n第二行\r\n"
        self.path.write_bytes(text.encode("utf-8-sig"))
        document = load_text(self.path)[0]
        self.assertEqual(document.page_content, text)
        self.assertEqual(document.metadata["line_end"], 2)

    def test_markdown_preserves_syntax_and_indentation(self):
        """Markdown 标题、链接、表格和代码块不被转换或清理。"""
        text = "# 标题\n\n[论文](https://example.com)\n|方法|结果|\n|---|---|\n```python\n    print('中文')\n```\n"
        for suffix in [".MD", ".markdown"]:
            with self.subTest(suffix=suffix):
                path = self.path.with_suffix(suffix)
                path.write_bytes(text.encode("utf-8"))
                document = load_text(path)[0]
                self.assertEqual(document.page_content, text)
                self.assertEqual(document.metadata["file_type"], suffix.lower())
                self.assertNotIn("page_number", document.metadata)

    def test_document_id_is_stable_and_changes(self):
        """内容指纹由原始字节生成，重复读取稳定，文本变化后更新。"""
        self.path.write_text("原始正文", encoding="utf-8")
        original_id = load_text(self.path)[0].metadata["doc_id"]
        self.assertEqual(load_text(self.path)[0].metadata["doc_id"], original_id)
        self.path.write_text("更新正文", encoding="utf-8")
        self.assertNotEqual(load_text(self.path)[0].metadata["doc_id"], original_id)

    def test_empty_or_whitespace_only_file(self):
        """空字节、仅 BOM 和纯空白都不视为有效内容。"""
        for data in [b"", b"\xef\xbb\xbf", b" \t\r\n"]:
            with self.subTest(data=data):
                self.path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, "没有有效内容"):
                    load_text(self.path)

    def test_invalid_utf8(self):
        """不静默替换错误字符，也不猜测 GBK 等其他编码。"""
        self.path.write_bytes("中文".encode("gbk"))
        with self.assertRaises(UnicodeDecodeError):
            load_text(self.path)

    def test_missing_file(self):
        """缺失文件明确报错。"""
        with self.assertRaises(FileNotFoundError):
            load_text(self.path)

    def test_unsupported_extension(self):
        """二进制格式不能被当作纯文本导入。"""
        path = self.path.with_suffix(".pdf")
        path.write_bytes(b"PDF")
        with self.assertRaisesRegex(ValueError, "不支持此文件格式"):
            load_text(path)


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


class TestImportFrontend(unittest.TestCase):
    """用 Streamlit 的真实上传组件驱动页面，隔离保存目录。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        config = {"app": {"name": "智能科研助理", "description": "导入测试"},
                  "paths": {"raw_documents": self.directory.name},
                  "importing": {"max_file_size_mb": 20}}
        configuration = patch("src.utils.config.load_config", return_value=config)
        configuration.start()
        self.addCleanup(configuration.stop)
        from streamlit.testing.v1 import AppTest
        app_path = Path(__file__).resolve().parents[1] / "src/frontend/app.py"
        # 新环境首次加载界面依赖较慢，避免默认 3 秒等待导致误报。
        self.app = AppTest.from_file(str(app_path), default_timeout=10).run()

    def test_batch_upload_progress_state_and_rerun(self):
        """实际操作上传/按钮，核验混合结果、完整进度和重跑不重复执行。"""
        app = self.app
        self.assertTrue(app.button(key="start_import").disabled)
        app.file_uploader[0].set_value([
            ("good.txt", "中文正文".encode("utf-8"), "text/plain"),
            ("bad.txt", b"\xff", "text/plain"),
        ]).run()
        app.button(key="start_import").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(list(app.dataframe[0].value["状态"]), ["成功", "失败"])
        self.assertEqual(app.session_state["import_progress"], {"completed": 2, "total": 2})
        self.assertEqual(app.get("progress")[0].proto.value, 100)
        self.assertTrue(app.button(key="start_import").disabled)
        self.assertFalse(app.button(key="retry_import").disabled)
        app.button(key="retry_import").click().run()
        self.assertEqual([t["attempts"] for t in app.session_state["import_tasks"]], [1, 2])
        app.run()
        self.assertEqual([t["attempts"] for t in app.session_state["import_tasks"]], [1, 2])

    def test_failed_upload_recovers_and_disables_retry(self):
        """页面通过失败按钮恢复成功，之后重试按钮禁用且文献存在。"""
        app = self.app
        app.file_uploader[0].set_value([("retry.txt", b"Retry", "text/plain")]).run()
        with patch("src.data_loader.load_document", side_effect=OSError("暂时失败")):
            app.button(key="start_import").click().run()
        self.assertEqual(app.session_state["import_tasks"][0]["status"], "failed")
        app.button(key="retry_import").click().run()
        self.assertFalse(app.exception)
        task = app.session_state["import_tasks"][0]
        self.assertEqual(task["status"], "success")
        self.assertEqual(task["attempts"], 2)
        self.assertEqual(task["error"], "")
        self.assertTrue(Path(task["path"]).is_file())
        self.assertTrue(app.button(key="retry_import").disabled)


if __name__ == "__main__":
    unittest.main()
