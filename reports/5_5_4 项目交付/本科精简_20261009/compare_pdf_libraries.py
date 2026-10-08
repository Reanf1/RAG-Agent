"""三库读取同一批冻结论文，记录速度、页数和既有证据锚点保留情况。"""

import argparse
from datetime import datetime
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform
import re
import time
import unicodedata

import PyPDF2
import pdfplumber
import pymupdf

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "reports/5_5_1 评测集构建/论文清单.json"


def normalize(text):
    """与冻结评测集一致：仅处理字体连字、换行断词和空白。"""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(re.sub(r"-\s*\n\s*", "", text).split()).casefold()


def extract_pages(path, library):
    """使用各库基础文本接口；不新增生产加载器或调参挑选答案。"""
    if library == "PyPDF2":
        with path.open("rb") as stream:
            return [page.extract_text() or "" for page in PyPDF2.PdfReader(stream).pages]
    if library == "pdfplumber":
        with pdfplumber.open(path) as pdf:
            return [page.extract_text() or "" for page in pdf.pages]
    with pymupdf.open(path) as pdf:
        return [page.get_text() for page in pdf]


def compare():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = []
    for library in ("PyPDF2", "pdfplumber", "PyMuPDF"):
        rows = []
        for paper in manifest["papers"]:
            path = ROOT / paper["local_path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == paper["pdf_sha256"]
            started = time.perf_counter()
            pages = extract_pages(path, library)
            elapsed = time.perf_counter() - started
            assert len(pages) == paper["pages"], (library, paper["id"])
            anchors = [{"id": item["id"], "page_number": item["page_number"],
                        "found": normalize(item["anchor"]) in normalize(pages[item["page_number"] - 1])}
                       for item in manifest["evidence"] if item["paper_id"] == paper["id"]]
            # 解析只读文件；计时不包含哈希与锚点核验。
            assert hashlib.sha256(path.read_bytes()).hexdigest() == paper["pdf_sha256"]
            rows.append({"paper_id": paper["id"], "sha256": paper["pdf_sha256"],
                         "pages": len(pages), "seconds": elapsed,
                         "text_characters": sum(map(len, pages)), "anchors": anchors})
            print(f"{library} {paper['id']}：{elapsed:.3f}秒", flush=True)
        results.append({"library": library, "version": version(library), "papers": rows,
                        "seconds": sum(row["seconds"] for row in rows),
                        "pages": sum(row["pages"] for row in rows),
                        "anchor_hits": sum(a["found"] for row in rows for a in row["anchors"]),
                        "anchor_total": sum(len(row["anchors"]) for row in rows)})
    return {"measured_at": datetime.now().astimezone().isoformat(),
            "platform": platform.platform(), "python": platform.python_version(),
            "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            "method": "各库预先导入；固定顺序、单次全量基础文本提取，不测OCR、渲染和表格识别；原PDF哈希不变。",
            "boundary": "43个既有锚点最初由PyMuPDF语料验证，不能当独立人工解析准确率；未命中可能由阅读顺序或空白差异造成。",
            "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError("新测量不能覆盖已存在的证据")
    output.write_text(json.dumps(compare(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
