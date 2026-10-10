"""5.1.2 文本分块策略：TestAcademicPDF。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import json
import tempfile
import unittest
import pymupdf
from src.data_loader.pdf_loader import load_pdf


class TestAcademicPDF(unittest.TestCase):
    """用实际绘制的论文版面验证阅读顺序、续表与公式，不 mock 解析器。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "academic.pdf"

    def draw_table(self, page, top, rows, caption="Table 1", three_lines=False):
        """绘制可重复的网格表或三线表，保留真正的文字和线段。"""
        if caption:
            page.insert_text((50, top - 15), caption)
        bottom = top + 30 * len(rows)
        columns = len(rows[0])
        if not three_lines:
            for index in range(columns + 1):
                x = 50 + 450 * index / columns
                page.draw_line((x, top), (x, bottom))
        ys = [top, top + 30, bottom] if three_lines else range(top, bottom + 1, 30)
        for y in ys:
            page.draw_line((50, y), (500, y))
        for row_index, row in enumerate(rows):
            for column, value in enumerate(row):
                page.insert_text((60 + 450 * column / columns, top + 20 + row_index * 30), value)

    def write_columns(self, page, top):
        """同一高度左右栏同时存在，朴素位置排序会交错读取。"""
        for index in range(3):
            for x, label in [(50, "LEFT"), (330, "RIGHT")]:
                page.insert_text((x, top + index * 25),
                                 f"{label}-{top}-{index} paragraph with many words", fontsize=9)

    def test_two_columns_with_full_width_heading(self):
        """先通栏标题，再完整左栏、完整右栏，最后页脚。"""
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            page.insert_text((190, 60), "ACADEMIC PAPER TITLE", fontsize=16)
            self.write_columns(page, 100)
            page.insert_text((270, 760), "FOOTER TEXT")
            pdf.save(self.path)
        text = load_pdf(self.path)[0].page_content
        self.assertLess(text.index("TITLE"), text.index("LEFT-100-0"))
        self.assertLess(text.index("LEFT-100-2"), text.index("RIGHT-100-0"))
        self.assertLess(text.index("RIGHT-100-2"), text.index("FOOTER"))


    def test_cross_page_table_merges_rows_and_keeps_row_sources(self):
        """相邻页同一续表合并，重复表头去重，每行记录真实来源页。"""
        with pymupdf.open() as pdf:
            self.draw_table(pdf.new_page(width=600,height=800), 650, [["Method","Score"],["A","90"],["B","95"]])
            self.draw_table(pdf.new_page(width=600,height=800), 70, [["Method","Score"],["C","96"],["D","97"]], "Table 1 (continued)")
            pdf.save(self.path)
        tables = [d for d in load_pdf(self.path) if d.metadata.get("content_type") == "table"]
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual(table.page_content.count("Method"), 1)
        self.assertIn("| C | 96 |", table.page_content)
        self.assertEqual((table.metadata["page_number"], table.metadata["page_end"]), (1,2))
        rows = json.loads(table.metadata["table_rows"])
        self.assertEqual([row["page_number"] for row in rows], [1,1,1,2,2])
        self.assertEqual(len(json.loads(table.metadata["table_regions"])), 2)


    def test_formula_symbols_and_scripts_are_preserved(self):
        """公式符号与上下标保持，不能把 x 的平方压成普通 x2。"""
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((60,80),"y = x",fontsize=14)
            page.insert_text((90,74),"2",fontsize=8)
            page.insert_text((100,80)," + a",fontsize=14)
            page.insert_text((124,85),"i",fontsize=8)
            page.insert_font(fontname="math",fontbuffer=pymupdf.Font("cjk").buffer)
            page.insert_text((60,120),"α + β = ∑ x / √n",fontname="math")
            pdf.save(self.path)
        document = load_pdf(self.path)[0]
        self.assertIn("x^{2}", document.page_content)
        self.assertIn("a_{i}", document.page_content)
        for symbol in ["α","β","∑","√"]:
            self.assertIn(symbol, document.page_content)
        layout = json.loads(document.metadata["formula_layout"])
        spans = [span for line in layout for span in line["spans"]]
        self.assertTrue(any(span["text"] == "2" and span["origin"][1] == 74 for span in spans))


if __name__ == "__main__":
    unittest.main()
