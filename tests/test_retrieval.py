"""模块一测试：使用临时文档验证文本与来源，不依赖模型或向量数据库。"""

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
