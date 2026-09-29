"""按已锁定的原文、问题和相关段落，准备 AI 论文双语检索基准。"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import split_documents
from src.data_loader.pdf_loader import load_pdf

MANIFEST = PROJECT_ROOT / "reports/Embedding论文双语评测集.json"
DATA_DIR = PROJECT_ROOT / "data/raw/embedding_papers"


def prepare_sample(download=False):
    """全部 PDF 分块都作为候选；哈希或标注变化时停止，不自动修正答案。"""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus, papers = [], {}
    for paper in manifest["papers"]:
        path = DATA_DIR / (paper["id"] + ".pdf")
        if download and not path.exists():
            request = urllib.request.Request(paper["pdf_url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == paper["pdf_sha256"], "PDF 版本变化"
        chunks = split_documents(load_pdf(path), **manifest["chunking"])
        assert len(chunks) == paper["chunk_count"], "解析或切分结果变化，请先复核基准"
        papers[paper["id"]] = (paper, chunks)
        for ordinal, chunk in enumerate(chunks):
            metadata = chunk.metadata
            corpus.append({"id": metadata["chunk_id"], "text": chunk.page_content,
                           "paper_id": paper["id"], "language": paper["language"],
                           "ordinal": ordinal, "page_number": metadata["page_number"],
                           "start_index": metadata["start_index"], "end_index": metadata["end_index"],
                           "content_type": metadata.get("content_type", "text")})
    queries = []
    for question in manifest["questions"]:
        paper, chunks = papers[question["paper_id"]]
        for location in question["relevant_chunks"]:
            metadata = chunks[location["ordinal"]].metadata
            for key in ("chunk_id", "page_number", "start_index", "end_index"):
                assert metadata[key] == location[key], "相关段落定位发生变化"
        for language in ("zh", "en"):
            queries.append({"id": question["id"] + "-" + language,
                            "pair_id": question["id"], "text": question[language],
                            "paper_id": paper["id"], "query_language": language,
                            "document_language": paper["language"],
                            "group": language + "->" + paper["language"],
                            "relevant_ids": [row["chunk_id"] for row in question["relevant_chunks"]]})
    sample = {"name": manifest["name"], "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
              "papers": manifest["papers"], "chunking": manifest["chunking"],
              "selection_rule": manifest["selection_rule"], "corpus": corpus, "queries": queries}
    payload = json.dumps(sample, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    path = DATA_DIR / "sample.json"
    path.write_bytes(payload)
    print("基准准备完成", len(corpus), "个候选", len(queries), "个问题",
          hashlib.sha256(payload).hexdigest(), flush=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="显式下载尚未准备的论文 PDF")
    prepare_sample(parser.parse_args().download)
