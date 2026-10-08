"""使用 PyMuPDF 加载论文正文、表格和公式位置，保留真实来源。"""

import hashlib
import json
import logging
import re
from pathlib import Path

import pymupdf
from langchain_core.documents import Document


def _read_lines(page, clip=None):
    """读取真实字符位置；小字号且偏离基线的字符保留上下标标记。"""
    flags = pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    lines = []
    for block in page.get_text("rawdict", flags=flags, clip=clip)["blocks"]:
        for line in block["lines"]:
            spans = []
            for span in line["spans"]:
                chars = span["chars"]
                # PyMuPDF 裁剪可能返回边缘相交字符；按字形基线排除邻行残片。
                if clip is not None:
                    chars = [c for c in chars if pymupdf.Point(c["origin"]) in pymupdf.Rect(clip)]
                if not chars:
                    continue
                visible = [c for c in chars if not c["c"].isspace()]
                spans.append({"text": "".join(c["c"] for c in chars),
                              "origin": (visible or chars)[0]["origin"],
                              "bbox": span["bbox"], "size": span["size"],
                              "flags": span["flags"]})
            if not spans:
                continue
            base = max(spans, key=lambda s: (s["size"], len(s["text"])))
            parts = []
            for span in spans:
                text = span["text"]
                offset = span["origin"][1] - base["origin"][1]
                # 仅处理水平行；旋转文字不按纵向坐标猜测上下标。
                if line["dir"] == (1.0, 0.0) and text.strip():
                    if span["flags"] & pymupdf.TEXT_FONT_SUPERSCRIPT:
                        text = "^{" + text.strip() + "}"
                    elif span["size"] < base["size"] * 0.85 and abs(offset) > base["size"] * 0.15:
                        text = ("^{" if offset < 0 else "_{") + text.strip() + "}"
                parts.append(text)
            lines.append({"bbox": line["bbox"], "text": "".join(parts).strip(), "spans": spans})
    return [line for line in lines if line["text"]]


def _reading_order(items, width):
    """常见居中分栏：分区域读完左栏再读右栏，通栏内容作为区域边界。"""
    middle = width / 2
    body = [item for item in items if "spans" in item and item["bbox"][2] - item["bbox"][0] >= width * 0.2]
    left = [item for item in body if item["bbox"][2] < middle]
    right = [item for item in body if item["bbox"][0] > middle]
    by_position = lambda item: (item["bbox"][1], item["bbox"][0])
    if len(left) < 2 or len(right) < 2:
        return sorted(items, key=by_position), False
    # 两栏需有纵向重叠和明显留白，避免误判单栏缩进或短公式。
    overlap = min(max(i["bbox"][3] for i in left), max(i["bbox"][3] for i in right)) - max(
        min(i["bbox"][1] for i in left), min(i["bbox"][1] for i in right))
    gap_left = max(i["bbox"][2] for i in left)
    gap_right = min(i["bbox"][0] for i in right)
    if overlap <= 0 or gap_right - gap_left < width * 0.02:
        return sorted(items, key=by_position), False
    split = (gap_left + gap_right) / 2
    wide = sorted([i for i in items if i["bbox"][0] < split < i["bbox"][2]], key=by_position)
    remaining = [i for i in items if not i["bbox"][0] < split < i["bbox"][2]]
    result = []
    for separator in wide + [None]:
        band = [i for i in remaining if separator is None or i["bbox"][1] < separator["bbox"][1]]
        result.extend(sorted(band, key=lambda i: (i["bbox"][0] >= split, *by_position(i))))
        remaining = [i for i in remaining if i not in band]
        if separator is not None:
            result.append(separator)
    return result, True


def _column_intervals(words):
    """合并文字在横向的覆盖区间，5 点内的小间隔视为同一列。"""
    intervals = []
    for word in sorted(words, key=lambda w: w[0]):
        if intervals and word[0] - intervals[-1][1] < 5:
            intervals[-1][1] = max(intervals[-1][1], word[2])
        else:
            intervals.append([word[0], word[2]])
    return intervals


def _three_line_table(page, clip, header_bottom):
    """用整表横向留白分列，避免将公式字符之间的小空隙当作新列。"""
    words = page.get_text("words", clip=clip)
    intervals = _column_intervals(words)
    header = [w for w in words if (w[1] + w[3]) / 2 < header_bottom]
    # 子表头列多于整表投影列，说明存在跨列多级表头，不能强行压成粗列。
    complex_header = any(len(_column_intervals([w for w in header
                         if left <= (w[0] + w[2]) / 2 <= right
                         and abs((w[1] + w[3] - anchor[1] - anchor[3]) / 2) < 2])) > 1
                         for left, right in intervals for anchor in header)
    if len(intervals) < 2 or complex_header:
        logging.getLogger(__name__).warning("第 %d 页三线表列边界不明确，保留原始正文与页码。", page.number + 1)
        return None
    edges = [clip.x0] + [(a[1] + b[0]) / 2 for a, b in zip(intervals, intervals[1:])] + [clip.x1]
    # 表头可多行；正文按主字号基线分行，边界取相邻基线中点。
    # 不能用字形框重叠分行：紧密排列时，上一行下伸部会碰到下一行。
    baselines = sorted(max(l["spans"], key=lambda s: s["size"])["origin"][1]
                       for l in _read_lines(page, clip))
    ys = []
    for y in baselines:
        if y >= header_bottom and (not ys or y - ys[-1] > 2):
            ys.append(y)
    boundaries = [header_bottom] + [(a + b) / 2 for a, b in zip(ys, ys[1:])] + [clip.y1]
    bands = [[clip.y0, header_bottom]] + list(zip(boundaries, boundaries[1:]))
    rows = []
    for top, bottom in bands:
        rows.append([" ".join(l["text"] for l in _read_lines(page, (left, top, right, bottom)))
                     for left, right in zip(edges, edges[1:])])
    return {"bbox": tuple(clip), "rows": rows, "columns": [a[0] for a in intervals]}


def _find_tables(page, lines):
    """网格表交给 PyMuPDF；有标题和三条横线的规则三线表用位置提取。"""
    drawings = page.get_drawings()
    # 立即复制普通数据，不保留依赖 Page/TableFinder 生命周期的对象。
    found = [{"bbox": t.bbox, "rows": t.extract(),
              "columns": [cell[0] if cell else None for cell in t.rows[0].cells]}
             for t in page.find_tables(strategy="lines_strict", paths=drawings).tables]
    captions = [l for l in lines if re.match(r"(?:Table|续?表)\s*\d+", l["text"], re.I)]
    # 距离使用完整标题段落，而非仅第一行；多行上置标题不能被下方另一标题抢占。
    caption_bounds = {}
    for block in page.get_text("blocks"):
        if sum(bool(re.match(r"(?:Table|续?表)\s*\d+", line.strip(), re.I))
               for line in block[4].splitlines()) != 1:
            continue
        for caption in captions:
            if pymupdf.Rect(caption["bbox"]) in pymupdf.Rect(block[:4]):
                # PDF有时把整张表和下置标题放进同一文本块；只取标题起始行之后。
                caption_bounds[id(caption)] = (*caption["bbox"][:3], block[3])
    rules = {}
    for drawing in drawings:
        rect = drawing["rect"]
        if rect.height <= 2 and rect.width >= page.cropbox.width * 0.3:
            key = (round(rect.x0 / 3) * 3, round(rect.x1 / 3) * 3)
            rules.setdefault(key, []).append((rect.y0 + rect.y1) / 2)
    for caption in captions:
        for (x0, x1), ys in rules.items():
            following = [c["bbox"][1] for c in captions if c["bbox"][1] > caption["bbox"][3]
                         and c["bbox"][0] < x1 and c["bbox"][2] > x0]
            limit = min(following) if following else page.cropbox.height
            ys = sorted(set(y for y in ys if caption["bbox"][3] <= y < limit))
            if len(ys) < 3 or ys[0] - caption["bbox"][3] > 65:
                continue
            clip = pymupdf.Rect(x0 - 1, ys[0], x1 + 1, ys[-1])
            if any(clip.intersects(pymupdf.Rect(t["bbox"])) for t in found):
                continue
            table = _three_line_table(page, clip, ys[1])
            if table is not None:
                found.append(table)
    caption_candidates = []
    for line in captions:
        bounds = caption_bounds.get(id(line), line["bbox"])
        distances = [(max(table["bbox"][1] - bounds[3], bounds[1] - table["bbox"][3]), index)
                     for index, table in enumerate(found)
                     if bounds[0] < table["bbox"][2] and bounds[2] > table["bbox"][0]]
        # 字形框的下伸部可能与表边界轻微重叠；标题中心仍须在表外。
        center = (line["bbox"][1] + line["bbox"][3]) / 2
        distances = sorted((max(0, distance), index) for distance, index in distances
                           if -4 <= distance <= 65 and not found[index]["bbox"][1] <= center <= found[index]["bbox"][3])
        if distances:
            caption_candidates.append((distances, line["text"]))
    # 先绑定只有一个候选的标题；相邻上置标题可转到下一表，不能挤掉已绑定标题。
    # 一个标题只绑定一张表、一张表只接收一个标题；同候选数优先距离更近者。
    assigned = {}
    for distances, text in sorted(caption_candidates, key=lambda item: (len(item[0]), item[0][0])):
        for _, index in distances:
            if index not in assigned:
                assigned[index] = text
                break
    fragments = []
    for table in sorted(found, key=lambda t: (t["bbox"][1], t["bbox"][0])):
        rows = [[cell or "" for cell in row] for row in table["rows"]]
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        if len(rows) < 2 or len(table["columns"]) < 2:
            continue
        caption = assigned.get(next(index for index, item in enumerate(found) if item is table), "")
        # 极稀疏的绘图网格不抽成表格，保留其原文本。
        if not caption and sum(bool(cell.strip()) for row in rows for cell in row) < len(rows) * len(table["columns"]) / 2:
            continue
        number = re.match(r"(?:Table|续?表)\s*(\d+(?:[.-]\d+)*)", caption, re.I)
        fragments.append({**table, "rows": rows, "caption": caption,
                          "number": number.group(1) if number else "",
                          "continued": bool(re.search(r"continued|续", caption, re.I)),
                          "page_number": page.number + 1, "height": page.cropbox.height})
    return fragments


def _join_tables(fragments, common):
    """只合并相邻页、页边位置与列结构一致且有续表证据的片段。"""
    groups = []
    for fragment in fragments:
        previous = groups[-1]["fragments"][-1] if groups else None
        repeat_header = bool(groups and fragment["rows"][0] == groups[-1]["rows"][0]["cells"])
        # 无标题的中间续页沿用首片段编号，不能把两张无编号表仅凭表头合并。
        number = groups[-1]["fragments"][0]["number"] if groups else ""
        same_number = bool(number and number == fragment["number"])
        columns_match = bool(previous and len(previous["columns"]) == len(fragment["columns"])
                             and all(a is not None and b is not None and abs(a - b) <= 4
                                     for a, b in zip(previous["columns"], fragment["columns"]))
                             and abs(previous["bbox"][2] - fragment["bbox"][2]) <= 4)
        number_conflict = bool(number and fragment["number"] and not same_number)
        merge = bool(previous and fragment["page_number"] == previous["page_number"] + 1
                     and previous["bbox"][3] >= previous["height"] * 0.75
                     and fragment["bbox"][1] <= fragment["height"] * 0.25
                     and columns_match and not number_conflict
                     and ((repeat_header and number) or (same_number and fragment["continued"])))
        rows = fragment["rows"][1:] if merge and repeat_header else fragment["rows"]
        if not merge:
            groups.append({"fragments": [], "rows": []})
        groups[-1]["fragments"].append(fragment)
        groups[-1]["rows"].extend({"cells": row, "page_number": fragment["page_number"]} for row in rows)
    documents = []
    for index, group in enumerate(groups):
        first, last = group["fragments"][0], group["fragments"][-1]
        # 有些PDF把多条模型数据合在一条网格行：各列等长、数字列逐行对齐时展开。
        # 表头、换行说明及列数不齐的行原样保留，不填补缺值或推测跨行对应。
        display_rows = []
        for position, row in enumerate(group["rows"]):
            cells = [cell.splitlines() for cell in row["cells"]]
            aligned = (position > 0 and len(cells[0]) > 1
                       and all(len(cell) == len(cells[0]) for cell in cells)
                       and all(re.fullmatch(r"[\d\s.,+−–\-×^()%@]+", value)
                               for cell in cells[1:] for value in cell))
            display_rows.extend([list(values) for values in zip(*cells)] if aligned else [row["cells"]])
        # 空单元格原样保留，不用相邻行的值补齐；转义 Markdown 分隔符。
        lines = ["| " + " | ".join(cell.replace("|", "\\|").replace("\n", "<br>") for cell in row) + " |"
                 for row in display_rows]
        lines.insert(1, "| " + " | ".join("---" for _ in group["rows"][0]["cells"]) + " |")
        metadata = {**common, "page": first["page_number"] - 1,
                    "page_number": first["page_number"], "page_end": last["page_number"],
                    "content_type": "table", "table_id": f"{common['doc_id']}:table:{index + 1}",
                    "table_rows": json.dumps(group["rows"], ensure_ascii=False),
                    "table_regions": json.dumps([{"page_number": f["page_number"], "bbox": f["bbox"]}
                                                 for f in group["fragments"]])}
        documents.append(Document(page_content=(first["caption"] + "\n" + "\n".join(lines)).strip(), metadata=metadata))
    return documents


def load_pdf(file_path: str | Path) -> list[Document]:
    """返回正文页与独立表格的 Document 列表，不执行 OCR 或公式语义重建。

    page 为从 0 开始的索引，page_number 为从 1 开始的物理页码。
    无文本页记录警告并跳过，全部无文本或需要密码时明确报错；
    损坏文件的解析异常直接向上传递，由后续批量导入层记录失败。
    """
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在或不是普通文件：{path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF 加载器不支持此文件格式：{path.suffix}")

    documents, fragments = [], []
    data = path.read_bytes()
    try:
        # 解析字节副本，不让MuPDF持有上传临时文件句柄；指纹与解析使用同一份内容。
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except pymupdf.FileDataError as error:
        raise pymupdf.FileDataError(f"PDF 文件损坏或不是有效的 PDF：{path.name}") from error
    # 上下文管理器确保加载成功或发生异常时都会关闭解析资源。
    with pdf:
        if not pdf.is_pdf:
            raise ValueError(f"文件内容不是 PDF：{path.name}")
        if pdf.needs_pass:
            raise ValueError(f"PDF 需要密码，请先解密后导入：{path.name}")

        # 同一文件的所有页共用内容指纹，重复加载时保持文档标识稳定。
        doc_id = hashlib.sha256(data).hexdigest()
        common = {"source": str(path), "source_file": path.name, "file_type": ".pdf",
                  "doc_id": doc_id, "total_pages": len(pdf)}
        for page in pdf:
            lines = _read_lines(page)
            if not lines:
                logging.getLogger(__name__).warning(
                    "%s 第 %d 页未提取到文本，已跳过；可能为空白或图片页，本加载器不执行 OCR。",
                    path.name, page.number + 1,
                )
                continue

            tables = _find_tables(page, lines)
            fragments.extend(tables)
            body = []
            for line in lines:
                rect = pymupdf.Rect(line["bbox"])
                center = (rect.tl + rect.br) / 2
                if not any(center in pymupdf.Rect(table["bbox"]) for table in tables):
                    body.append(line)
            body.extend({"bbox": t["bbox"], "text": f"[表格：原文第 {page.number + 1} 页，数据见独立表格内容]"}
                        for t in tables)
            images = [{"page_number": page.number + 1, "bbox": image["bbox"]}
                      for image in page.get_image_info()]
            body.extend({"bbox": image["bbox"], "text": f"[图像区域 {i + 1}：原文第 {page.number + 1} 页]"}
                        for i, image in enumerate(images))
            ordered, two_columns = _reading_order(body, page.cropbox.width)
            metadata = {**common, "page": page.number, "page_number": page.number + 1}
            if two_columns:
                metadata["layout"] = "two_column"
            # 二维公式不猜测 LaTeX：保留本页字符位置、字体大小与原 PDF 来源。
            if any(re.search(r"[=+∑∫√±×÷≤≥α-ωΑ-Ω]|\^\{|_\{", line["text"]) for line in lines):
                metadata["formula_layout"] = json.dumps(lines, ensure_ascii=False)
            if images:
                metadata["image_regions"] = json.dumps(images)
            text = "\n".join(item["text"] for item in ordered).strip()
            if "\ufffd" in text:
                logging.getLogger(__name__).warning("%s 第 %d 页存在无法映射的字符，请按原 PDF 核对公式。", path.name, page.number + 1)
            documents.append(Document(page_content=text, metadata=metadata))
        documents.extend(_join_tables(fragments, common))

    if not documents:
        raise ValueError(f"PDF 未提取到文本，请检查是否为空白或扫描件；本加载器不执行 OCR：{path.name}")
    return documents
