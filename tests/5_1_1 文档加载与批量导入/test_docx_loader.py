"""5.1.1 文档加载与批量导入：TestDOCXLoader。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hashlib
import os
import tempfile
import unittest
from docx import Document as WordDocument
from docx.opc.exceptions import PackageNotFoundError
from langchain_core.documents import Document
from src.data_loader.docx_loader import load_docx


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


if __name__ == "__main__":
    unittest.main()
