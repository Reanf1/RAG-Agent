"""模块一测试：使用临时 PDF 验证文本与来源，不依赖模型或向量数据库。"""

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

# 直接运行测试文件时，按文件位置加入项目根目录，不依赖当前工作目录。
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader.pdf_loader import load_pdf


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


if __name__ == "__main__":
    unittest.main()
