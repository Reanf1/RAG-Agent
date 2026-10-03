"""5.1.1 文档加载与批量导入：TestTextLoader。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hashlib
import os
import tempfile
import unittest
from src.data_loader.text_loader import load_text


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
