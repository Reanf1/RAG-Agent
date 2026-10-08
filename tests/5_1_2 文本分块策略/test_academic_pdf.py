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
from src.data_loader import batch_import, create_import_tasks


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

    def test_full_width_separator_between_column_sections(self):
        """页中通栏内容将双栏分为上下区域，避免跨区域串读。"""
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            self.write_columns(page, 100)
            page.insert_text((180, 220), "FULL WIDTH SECTION HEADING", fontsize=14)
            self.write_columns(page, 270)
            pdf.save(self.path)
        text = load_pdf(self.path)[0].page_content
        expected = ["LEFT-100-2", "RIGHT-100-0", "HEADING", "LEFT-270-0", "RIGHT-270-0"]
        self.assertEqual([text.index(token) for token in expected], sorted(text.index(token) for token in expected))

    def test_indented_single_column_is_not_reordered(self):
        """上下分离的缩进段落不能被误当作同时并排的两栏。"""
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            for y, x, label in [(100,330,"FIRST"),(125,330,"SECOND"),(300,50,"THIRD"),(325,50,"FOURTH")]:
                page.insert_text((x,y), label + " paragraph with enough words", fontsize=9)
            pdf.save(self.path)
        text = load_pdf(self.path)[0].page_content
        self.assertLess(text.index("FIRST"), text.index("THIRD"))

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

    def test_continuation_without_repeated_header(self):
        """明确同编号续表可无重复表头，下一页首行数据不可丢失。"""
        with pymupdf.open() as pdf:
            self.draw_table(pdf.new_page(width=600,height=800),650,[["Method","Score"],["A","90"],["B","91"]])
            self.draw_table(pdf.new_page(width=600,height=800),70,[["C","92"],["D","93"]],"Table 1 (continued)")
            pdf.save(self.path)
        tables = [d for d in load_pdf(self.path) if d.metadata.get("content_type") == "table"]
        self.assertEqual(len(tables), 1)
        self.assertIn("| C | 92 |", tables[0].page_content)
        self.assertIn("| D | 93 |", tables[0].page_content)

    def test_different_numbered_tables_do_not_merge(self):
        """几何位置和表头相同也不能合并不同编号的独立表。"""
        with pymupdf.open() as pdf:
            self.draw_table(pdf.new_page(width=600,height=800),650,[["Method","Score"],["A","90"]],"Table 1")
            self.draw_table(pdf.new_page(width=600,height=800),70,[["Method","Score"],["B","91"]],"Table 2")
            pdf.save(self.path)
        self.assertEqual(sum(d.metadata.get("content_type") == "table" for d in load_pdf(self.path)), 2)

    def test_below_captions_do_not_attach_to_the_next_table(self):
        """相邻网格表下置标题分别绑定，不能把Table5赋给Table6。"""
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            self.draw_table(page, 100, [["Method", "Score"], ["FIRST", "90"]], caption="")
            page.insert_text((50, 176), "Table 5: first results")
            self.draw_table(page, 195, [["Method", "Score"], ["SECOND", "95"]], caption="")
            page.insert_text((50, 271), "Table 6: second results")
            pdf.save(self.path)
        tables = [d for d in load_pdf(self.path) if d.metadata.get("content_type") == "table"]
        self.assertEqual([t.page_content.splitlines()[0] for t in tables],
                         ["Table 5: first results", "Table 6: second results"])

    def test_multiline_top_caption_wins_over_next_tables_caption(self):
        """上置标题最后一行最接近本表，不能只按第一行距离改绑下方标题。"""
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            page.insert_text((50, 60), "Table 3: model results\nTraining details line\nEvaluation details line", fontsize=11)
            self.draw_table(page, 100, [["Method", "Score"], ["CAIT", "90"]], caption="")
            page.insert_text((50, 177), "Table 4: ablation")
            self.draw_table(page, 200, [["Method", "Score"], ["OTHER", "95"]], caption="")
            pdf.save(self.path)
        tables = [d for d in load_pdf(self.path) if d.metadata.get("content_type") == "table"]
        self.assertEqual([t.page_content.splitlines()[0] for t in tables],
                         ["Table 3: model results", "Table 4: ablation"])

    def test_nonadjacent_or_incompatible_tables_do_not_merge(self):
        """页码断开、表头或列结构不同、非页边续接均分别保留。"""
        for middle_page, top, rows in [(True,70,[["Method","Score"],["B","91"]]),
                                      (False,70,[["Data","Count"],["B","91"]]),
                                      (False,70,[["Method","Score","Time"],["B","91","2"]]),
                                      (False,350,[["Method","Score"],["B","91"]])]:
            with self.subTest(middle_page=middle_page,top=top,rows=rows):
                with pymupdf.open() as pdf:
                    self.draw_table(pdf.new_page(width=600,height=800),650,[["Method","Score"],["A","90"]])
                    if middle_page:
                        pdf.new_page(width=600,height=800).insert_text((50,100),"Intermediate page")
                    self.draw_table(pdf.new_page(width=600,height=800),top,rows,"Table 1")
                    pdf.save(self.path)
                self.assertEqual(sum(d.metadata.get("content_type") == "table" for d in load_pdf(self.path)), 2)

    def test_three_line_table_and_empty_cells(self):
        """三线表在规则围出的区域内识别，空单元格不从相邻行填值。"""
        with pymupdf.open() as pdf:
            self.draw_table(pdf.new_page(width=600,height=800),150,[["Method","Score"],["A","90"],["B",""],["C","95"]],three_lines=True)
            pdf.save(self.path)
        tables = [d for d in load_pdf(self.path) if d.metadata.get("content_type") == "table"]
        self.assertEqual(len(tables), 1)
        self.assertIn("| B |  |", tables[0].page_content)
        self.assertEqual(json.loads(tables[0].metadata["table_rows"])[2]["cells"], ["B", ""])

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

    def test_multiple_table_styles_on_same_page(self):
        """同页网格表与三线表分别保留，不把横线范围或单元格混在一起。"""
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=600,height=800)
            self.draw_table(page,150,[["Method","Score"],["GRID","90"]],"Table 1")
            self.draw_table(page,400,[["Method","Score"],["PLAIN","95"],["OTHER","96"]],"Table 2",three_lines=True)
            pdf.save(self.path)
        tables=[d for d in load_pdf(self.path) if d.metadata.get("content_type")=="table"]
        self.assertEqual(len(tables),2)
        self.assertIn("GRID",tables[0].page_content)
        self.assertNotIn("PLAIN",tables[0].page_content)
        self.assertIn("PLAIN",tables[1].page_content)

    def test_three_line_formula_cells_and_header_words(self):
        """多词表头和公式中的间隔不拆成虚假的列。"""
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=600,height=800)
            self.draw_table(page,150,[["Layer Type","Complexity per Layer"],["Attention","O(n * d)"],["Recurrent","O(n * d * d)"]],three_lines=True)
            pdf.save(self.path)
        table=next(d for d in load_pdf(self.path) if d.metadata.get("content_type")=="table")
        self.assertEqual(len(json.loads(table.metadata["table_rows"])[0]["cells"]),2)
        self.assertIn("Complexity per Layer",table.page_content)
        self.assertIn("O(n * d * d)",table.page_content)

    def test_chinese_continuation_and_unnumbered_tables(self):
        """中文续表无重复表头时可合并；无编号的独立表不凭同表头合并。"""
        for numbered in [True,False]:
            with self.subTest(numbered=numbered):
                with pymupdf.open() as pdf:
                    first=pdf.new_page(width=600,height=800)
                    self.draw_table(first,650,[["Method","Score"],["A","90"]],"")
                    if numbered:
                        first.insert_text((50,635),"表 1",fontname="china-s")
                    second=pdf.new_page(width=600,height=800)
                    self.draw_table(second,70,[["B","91"],["C","92"]] if numbered else [["Method","Score"],["B","91"]],"")
                    if numbered:
                        second.insert_text((50,55),"续表 1",fontname="china-s")
                    pdf.save(self.path)
                tables=[d for d in load_pdf(self.path) if d.metadata.get("content_type")=="table"]
                self.assertEqual(len(tables),1 if numbered else 2)
                self.assertIn("B",tables[-1].page_content)

    def test_dense_formula_rows_do_not_leak_into_neighbors(self):
        """上标边缘与邻行相交时，不能向表头或邻行复制残留字符。"""
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=600,height=800)
            page.insert_text((50,85),"Table 1")
            for y in [100,120,158]:page.draw_line((50,y),(500,y))
            page.insert_text((60,115),"Method",fontsize=11)
            page.insert_text((300,115),"Complexity",fontsize=11)
            for y,label in [(130,"Attention"),(142,"Recurrent")]:
                page.insert_text((60,y),label,fontsize=11)
                page.insert_text((300,y),"x",fontsize=11)
                page.insert_text((306,y-4),"2",fontsize=7)
            pdf.save(self.path)
        table=next(d for d in load_pdf(self.path) if d.metadata.get("content_type")=="table")
        rows=json.loads(table.metadata["table_rows"])
        self.assertEqual([row["cells"] for row in rows],[["Method","Complexity"],["Attention","x^{2}"],["Recurrent","x^{2}"]])

    def test_ambiguous_three_line_header_keeps_original_text(self):
        """父表头横跨两个子列时不输出错误结构，正文和原文位置仍保留。"""
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=600,height=800)
            page.insert_text((50,85),"Table 1")
            for y in [100,145,220]:page.draw_line((50,y),(500,y))
            page.insert_text((60,120),"Method")
            page.insert_text((285,115),"Combined long parent heading")
            page.insert_text((285,138),"LEFT")
            page.insert_text((400,138),"RIGHT")
            for y,label in [(165,"A"),(190,"B")]:
                page.insert_text((60,y),label)
                page.insert_text((285,y),"90")
                page.insert_text((400,y),"95")
            pdf.save(self.path)
        with self.assertLogs("src.data_loader.pdf_loader",level="WARNING"):
            documents=load_pdf(self.path)
        self.assertFalse(any(d.metadata.get("content_type")=="table" for d in documents))
        self.assertIn("LEFT",documents[0].page_content)
        self.assertIn("RIGHT",documents[0].page_content)

    def test_three_page_table_keeps_first_caption_identity(self):
        """中间续页不重复标题时，仍保留整张表的编号和第三页数据。"""
        with pymupdf.open() as pdf:
            self.draw_table(pdf.new_page(width=600,height=800),650,[["Method","Score"],["A","90"],["B","91"]])
            self.draw_table(pdf.new_page(width=600,height=800),70,[["Method","Score"]]+[[f"M{i}",str(i)] for i in range(22)],"")
            self.draw_table(pdf.new_page(width=600,height=800),70,[["Method","Score"],["FINAL","99"]],"")
            pdf.save(self.path)
        tables=[d for d in load_pdf(self.path) if d.metadata.get("content_type")=="table"]
        self.assertEqual(len(tables),1)
        self.assertEqual(tables[0].metadata["page_end"],3)
        self.assertIn("FINAL",tables[0].page_content)

    def test_fraction_layout_retains_original_positions(self):
        """不猜测二维分式的 LaTeX，保留分子/分母位置以回看原文。"""
        with pymupdf.open() as pdf:
            page=pdf.new_page()
            page.insert_text((60,130),"f =",fontsize=12)
            page.insert_text((95,119),"1",fontsize=10)
            page.draw_line((92,125),(119,125))
            page.insert_text((95,140),"1 + x",fontsize=10)
            pdf.save(self.path)
        document = load_pdf(self.path)[0]
        spans = [s for line in json.loads(document.metadata["formula_layout"]) for s in line["spans"]]
        self.assertTrue(any(s["text"] == "1" and s["origin"][1] == 119 for s in spans))
        self.assertTrue(any("1 + x" in s["text"] and s["origin"][1] == 140 for s in spans))

    def test_aligned_multiline_numeric_rows_keep_model_units_and_values_together(self):
        """明确模拟PDF网格提取结果；原始单元格仍留在元数据供原页核对。"""
        from src.data_loader.pdf_loader import _join_tables
        rows = [['Model', 'Params (M)', 'FLOPs (B)', 'Accuracy (%)'],
                ['Small\nLarge', '12.0\n270.9', '9.6\n173.3', '80.4\n84.9']]
        fragment = {'rows': rows, 'columns': [0, 50, 100, 150], 'bbox': [0, 20, 200, 100],
                    'page_number': 1, 'height': 800, 'number': '1', 'caption': 'Table 1. Models', 'continued': False}
        table = _join_tables([fragment], {'doc_id': 'probe'})[0]
        self.assertIn('| Large | 270.9 | 173.3 | 84.9 |', table.page_content)
        self.assertIn('| Small | 12.0 | 9.6 | 80.4 |', table.page_content)
        self.assertEqual(json.loads(table.metadata['table_rows'])[1]['cells'], rows[1])

    def test_unequal_multiline_columns_are_not_guessed_into_rows(self):
        from src.data_loader.pdf_loader import _join_tables
        rows = [['Model', 'Value'], ['Small\nLarge', '12.0']]
        fragment = {'rows': rows, 'columns': [0, 100], 'bbox': [0, 20, 200, 100],
                    'page_number': 1, 'height': 800, 'number': '1', 'caption': 'Table 1. Models', 'continued': False}
        table = _join_tables([fragment], {'doc_id': 'probe'})[0]
        self.assertIn('| Small<br>Large | 12.0 |', table.page_content)

    def test_image_formula_is_locatable_after_batch_save(self):
        """混合文本页的公式图片保留原文定位，批量保存后仍可裁剪查看。"""
        with pymupdf.open() as picture:
            image_page=picture.new_page(width=180,height=50)
            image_page.insert_text((10,30),"y = (a+b)/c",fontsize=16)
            image=image_page.get_pixmap().tobytes("png")
        with pymupdf.open() as pdf:
            page=pdf.new_page()
            page.insert_text((60,80),"Image equation follows:")
            page.insert_image(pymupdf.Rect(60,100,240,150),stream=image)
            pdf.save(self.path)
        tasks=create_import_tasks([("formula.pdf",self.path.read_bytes())])
        list(batch_import(tasks, Path(self.directory.name)/"raw"))
        self.assertEqual(tasks[0]["status"],"success")
        document=tasks[0]["documents"][0]
        region=json.loads(document.metadata["image_regions"])[0]
        self.assertIn("图像区域",document.page_content)
        with pymupdf.open(document.metadata["source"]) as saved:
            crop=saved[region["page_number"]-1].get_pixmap(clip=region["bbox"])
            self.assertEqual((crop.width,crop.height),(180,50))
            self.assertGreater(len(crop.tobytes("png")),100)


if __name__ == "__main__":
    unittest.main()
