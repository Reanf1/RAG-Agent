"""在课程模板的副本中填入真实项目报告，原始DOCX保持不变。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
from urllib.parse import unquote
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.text.run import Run

ROOT = Path(__file__).resolve().parents[2]


def font(run, name="仿宋", size=14, bold=False):
    """中文字体名称保留模板约定，渲染字体替代仅在临时环境设置。"""
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages", type=Path, help="视觉核验后的标题实际页号JSON")
    args = parser.parse_args()
    markdown = (ROOT / "docs/课程报告.md").read_text(encoding="utf-8")
    doc = Document(ROOT / "docs/项目交付模板.docx")
    original = list(doc.element.body)
    for element in original:
        if element.tag != qn("w:sectPr"):
            doc.element.body.remove(element)
    # 不重建模板包：section、主题、页眉页脚、关系及原样式均保留。
    section = doc.sections[0]
    width = section.page_width - section.left_margin - section.right_margin
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "仿宋", Pt(14)
    normal.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "仿宋")
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.keep_with_next = False
    normal.paragraph_format.keep_together = False
    normal.paragraph_format.page_break_before = False
    snap = OxmlElement('w:snapToGrid'); snap.set(qn('w:val'), '0')
    normal.element.get_or_add_pPr().append(snap)
    # 模板未提供正文页码，追加简单PAGE域供目录页号核对。
    footer = section.footer.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.paragraph_format.line_spacing = 1
    footer.paragraph_format.space_after = Pt(0)
    footer.paragraph_format.space_before = Pt(0)
    page_field = OxmlElement('w:fldSimple'); page_field.set(qn('w:instr'), 'PAGE')
    footer._p.append(page_field)
    for level, size in ((1, 16), (2, 14), (3, 14)):
        name = f"Heading {level}"
        style = doc.styles[name] if name in doc.styles else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        style.font.name, style.font.size, style.font.bold = "黑体", Pt(size), True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "黑体")
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.line_spacing = 1.5

    def clone(index):
        element = deepcopy(original[index])
        doc.element.body.insert(len(doc.element.body) - 1, element)
        # 模板的手动换页统一改成明确的页面边界，避免空白页。
        for br in element.findall('.//w:br', namespaces=element.nsmap):
            if br.get(qn('w:type')) == 'page': br.getparent().remove(br)
        return element

    # 考核表仍由教师填写，日期沿用模板，不当作实际截止日期。
    for index in (2, 3, 4, 5, 6, 7): clone(index)
    for _ in range(5): doc.add_paragraph("________________________________________")
    for index in (20, 21, 22): clone(index)
    if "Title" not in doc.styles:
        doc.styles.add_style("Title", WD_STYLE_TYPE.PARAGRAPH)
    title = doc.add_paragraph("生产实习课程报告", style="Title")
    title.paragraph_format.page_break_before = True
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(48)
    title.paragraph_format.space_after = Pt(60)
    font(title.runs[0], "隶书", 30, True)
    topic = doc.add_paragraph("题目：智能科研助理\n基于 RAG + Agent 的论文知识库问答系统")
    topic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    font(topic.runs[0], "黑体", 16, True)
    for index in range(1, 5):
        p = doc.add_paragraph(f"成员 {index}：班级________ 学号________ 姓名________")
        p.paragraph_format.space_before = Pt(14)
        font(p.runs[0], "黑体", 14)
    doc.add_paragraph("资料截止：2026年10月7日；个人信息与真实贡献比例待填写。")

    headings = [(line.count('#'), line.lstrip('#').strip()) for line in markdown.splitlines()
                if line.startswith('## ') or line.startswith('### ')]
    page_map = json.loads(args.pages.read_text(encoding="utf-8")) if args.pages else {}
    toc = doc.add_paragraph("目录")
    toc.paragraph_format.page_break_before = True
    toc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    font(toc.runs[0], "黑体", 22, True)
    # 由实际渲染页号生成静态目录，标题书签可在Word中跳转。
    anchors = {text: f"section_{i}" for i, (_, text) in enumerate(headings)}
    for depth, text in headings:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.left_indent = Cm(.5 if depth == 3 else 0)
        link = OxmlElement('w:hyperlink'); link.set(qn('w:anchor'), anchors[text])
        run = OxmlElement('w:r'); contents = OxmlElement('w:t')
        contents.text = f"{text} …… {page_map.get(text, '待核定')}"
        run.append(contents); font(Run(run, p), size=10)
        link.append(run); p._p.append(link)

    def paragraph(text, *, code=False):
        p = doc.add_paragraph(re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text))
        p.paragraph_format.widow_control = True
        if code:
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(0)
            font(p.runs[0], "黑体", 8.5)
        else:
            p.paragraph_format.first_line_indent = Cm(.85)
        return p

    lines, index, code = markdown.splitlines(), 0, False
    introduction = lines[2]
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith('<!--') or line.startswith('# '): continue
        if line == introduction:
            continue
        if line.startswith('```'):
            code = not code; continue
        if code:
            paragraph(lines[index - 1], code=True); continue
        if line.startswith('#'):
            depth, text = len(line) - len(line.lstrip('#')), line.lstrip('#').strip()
            p = doc.add_heading(text, level=min(depth - 1, 3))
            if depth == 2: p.paragraph_format.page_break_before = True
            if text == '一 项目概述': paragraph(introduction)
            if text in anchors:
                bookmark = OxmlElement('w:bookmarkStart')
                bookmark.set(qn('w:id'), str(headings.index((depth, text)) + 1))
                bookmark.set(qn('w:name'), anchors[text])
                p._p.insert(1 if p._p.pPr is not None else 0, bookmark)  # 段落属性按OOXML顺序保持首项。
                end = OxmlElement('w:bookmarkEnd'); end.set(qn('w:id'), bookmark.get(qn('w:id'))); p._p.append(end)
        elif line.startswith('|'):
            rows = [line]
            while index < len(lines) and lines[index].strip().startswith('|'):
                rows.append(lines[index].strip()); index += 1
            cells = [[s.strip() for s in row.strip('|').split('|')] for row in rows
                     if not re.fullmatch(r'[|\s:\-]+', row)]
            table = doc.add_table(rows=0, cols=len(cells[0]))
            table.style = 'Table Grid'
            table.autofit = False
            widths = {3: [3.2, 5.2, 6.25], 4: [4.6, 1.75, 1.75, 6.55],
                      5: [3.3, 3.3, 3.3, 2.3, 2.45], 6: [2.75, 2.7, 1.3, 2.4, 3.1, 2.4]}[len(cells[0])]
            for col, cm in zip(table.columns, widths): col.width = Cm(cm)
            for n, values in enumerate(cells):
                row = table.add_row()
                properties = row._tr.get_or_add_trPr()
                properties.append(OxmlElement('w:cantSplit'))
                if n == 0: properties.append(OxmlElement('w:tblHeader'))
                for cell, value, cm in zip(row.cells, values, widths):
                    cell.width = Cm(cm); cell.text = value
                    for p in cell.paragraphs:
                        # 表内文字左对齐，避免模板的两端对齐把短词拉开。
                        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                        p.paragraph_format.line_spacing = 1.15
                        p.paragraph_format.space_after = Pt(4)
                        p.paragraph_format.space_before = Pt(4)
                        for run in p.runs: font(run, size=11, bold=n == 0)
            doc.add_paragraph()
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[([^]]*)\]\((.*)\)', line)
            image_path = (ROOT / 'docs' / unquote(match.group(2))).resolve()
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            # 截图保留完整画面并给图注；高图限高以免溢出正文页。
            from PIL import Image
            with Image.open(image_path) as image:
                ratio = image.height / image.width
            use_width = min(width, int(Cm(15) / ratio))
            p.add_run().add_picture(str(image_path), width=use_width)
            p.paragraph_format.keep_with_next = True
        else:
            paragraph(line)
    setting = OxmlElement('w:updateFields'); setting.set(qn('w:val'), 'true')
    doc.settings.element.append(setting)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.output)
    print(f"已从模板副本生成 {args.output}；成员资料与用户质量审核保持待填。")


if __name__ == '__main__':
    main()
