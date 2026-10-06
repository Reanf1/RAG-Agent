"""保存助手逐条初评用的原文证据；只提取与核验，不自动计算质量分。"""

from hashlib import sha256
import json
from pathlib import Path
import re

import fitz


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "reports/5_5_1 评测集构建/论文清单.json").read_text())
questions = json.loads((ROOT / "reports/评测集.json").read_text())
answers = json.loads((HERE / "人工评分180条来源.json").read_text())
pages = {}
for question in questions:
    for item in question["evidence"]:
        pages.setdefault(item["paper_id"], set()).add(item["page_number"])
for row in answers["rows"]:
    # 最终答案中可定位的文件/页码也核对；孤立参考编号不从内部轨迹补全。
    text = row["answer"].replace("\\_", "_")
    for paper in manifest["papers"]:
        for match in re.finditer(re.escape(paper["id"] + ".pdf") + r"[^\n]{0,25}?第(\d+)页", text):
            pages.setdefault(paper["id"], set()).add(int(match.group(1)))
records = []
for paper in manifest["papers"]:
    path = ROOT / paper["local_path"]
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != paper["pdf_sha256"]:
        raise ValueError(f"原文指纹变化：{paper['id']}")
    with fitz.open(path) as pdf:
        for number in sorted(pages[paper["id"]]):
            if not 1 <= number <= len(pdf):
                raise ValueError(f"引用页码越界：{paper['id']}:{number}")
            records.append({"paper_id": paper["id"], "physical_page": number,
                            "source_path": paper["local_path"], "pdf_sha256": digest,
                            "text": pdf[number - 1].get_text(sort=True)})
result = {"purpose": "助手初评的原文与最终答案引用核验，不是自动打分",
          "page_numbering": "PDF物理页码，从1开始", "pages": records}
(HERE / "助手初评原文证据_20261006.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已核验12份PDF指纹，提取{len(records)}页原文；未修改原资料。")
