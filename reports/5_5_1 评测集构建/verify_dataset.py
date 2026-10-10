"""校验固定论文和四类问答，准备只含论文原文的离线候选语料。"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import unicodedata
from urllib.request import Request, urlopen

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.chunking import split_documents
from src.data_loader.pdf_loader import load_pdf

MANIFEST = ROOT / "reports/5_5_1 评测集构建/论文清单.json"
DATASET = ROOT / "reports/评测集.json"
CORPUS = ROOT / "data/raw/evaluation_vision_transformers/corpus.json"
CATEGORIES = {"fact", "comparison", "synthesis", "reasoning"}


def normalize(text):
    """只合并换行断词、空白和字体连字，不纠正或改写论文内容。"""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(re.sub(r"-\s*\n\s*", "", text).split()).casefold()


def prepare_and_verify(download=False):
    """校验失败即停止；不根据模型输出修改答案或重新选择证据。"""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    questions = json.loads(DATASET.read_text(encoding="utf-8"))
    papers = {paper["id"]: paper for paper in manifest["papers"]}
    evidence = {item["id"]: item for item in manifest["evidence"]}
    assert 10 <= len(papers) <= 20 and len(papers) == len(manifest["papers"])
    assert len(questions) >= 50 and len(questions) == manifest["question_count"]
    assert len({q["id"] for q in questions}) == len(questions)
    assert len({normalize(q["question"]) for q in questions}) == len(questions)
    assert set(q["category"] for q in questions) == CATEGORIES
    assert set(q["language"] for q in questions) == {"zh", "en"}
    assert len(evidence) == len(manifest["evidence"])
    used_papers, used_evidence = set(), set()
    for question in questions:
        assert question["question"].strip() and question["reference_answer"].strip()
        assert question["answer_points"] and all(p.strip() for p in question["answer_points"])
        assert question["answer_basis"].strip() and question["evidence"]
        assert len(question["paper_ids"]) == len(set(question["paper_ids"]))
        assert set(question["paper_ids"]) <= papers.keys()
        assert set(question["paper_ids"]) == {row["paper_id"] for row in question["evidence"]}
        if question["category"] in {"comparison", "synthesis"}:
            assert len(question["paper_ids"]) >= 2
        for location in question["evidence"]:
            expected = evidence[location["evidence_id"]]
            for key in ("paper_id", "page_number", "section"):
                assert location[key] == expected[key], question["id"]
            used_evidence.add(location["evidence_id"])
        used_papers.update(question["paper_ids"])
    assert used_papers == papers.keys() and used_evidence == evidence.keys()

    corpus, paper_checks, anchor_checks = [], [], []
    for paper in papers.values():
        path = ROOT / paper["local_path"]
        if not path.exists() and download:
            path.parent.mkdir(parents=True, exist_ok=True)
            request = Request(paper["pdf_url"], headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=60) as response:
                payload = response.read()
            # 下载内容通过固定哈希后才能写入，不能静默接受更新的论文版本。
            assert hashlib.sha256(payload).hexdigest() == paper["pdf_sha256"], paper["id"]
            path.write_bytes(payload)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == paper["pdf_sha256"] == paper["doc_id"], paper["id"]
        documents = load_pdf(path)
        chunks = split_documents(documents, **manifest["chunking"])
        assert documents and chunks and all(d.metadata["doc_id"] == digest for d in chunks)
        with pymupdf.open(path) as pdf:
            assert len(pdf) == paper["pages"]
            for item in (e for e in evidence.values() if e["paper_id"] == paper["id"]):
                page = item["page_number"]
                assert type(page) is int and 1 <= page <= len(pdf)
                anchor = normalize(item["anchor"])
                assert anchor in normalize(pdf[page - 1].get_text()), item["id"]
                matched = [d for d in documents if d.metadata["page_number"] == page
                           and anchor in normalize(d.page_content)]
                assert matched, "加载器未保留证据锚点：" + item["id"]
                matching_ids = [d.metadata["chunk_id"] for d in chunks
                                if d.metadata["page_number"] == page
                                and anchor in normalize(d.page_content)]
                assert matching_ids, "分块后未找到证据锚点：" + item["id"]
                anchor_checks.append({"evidence_id": item["id"], "paper_id": paper["id"],
                                      "page_number": page, "anchor_verified": True,
                                      "matching_chunk_ids": matching_ids})
        # 保留评测所需正文与来源；每页大型版面元数据可按PDF重新读取，避免逐块重复。
        for chunk in chunks:
            corpus.append({"paper_id": paper["id"], "id": chunk.metadata["chunk_id"],
                           "text": chunk.page_content, "source_pdf": paper["local_path"],
                           "metadata": {key: chunk.metadata[key] for key in
                                        ("chunk_id", "doc_id", "source_file", "page_number",
                                         "page_end", "start_index", "end_index", "content_type")
                                        if key in chunk.metadata}})
        paper_checks.append({"paper_id": paper["id"], "pdf_sha256": digest,
                             "pages": paper["pages"], "loaded_units": len(documents),
                             "chunks": len(chunks), "passed": True})
        print(f"{paper['id']}：{paper['pages']}页，{len(chunks)}块，原文及证据定位通过", flush=True)

    assert len({row["id"] for row in corpus}) == len(corpus)
    # 语料只包含原始论文块，不混入问题、参考答案或评分要点。
    payload = {"manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
               "chunking": manifest["chunking"], "corpus": corpus}
    serialized = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if CORPUS.exists():
        assert CORPUS.read_bytes() == serialized, "既有语料变化，请复核版本并使用新目录"
    else:
        CORPUS.write_bytes(serialized)
    return {"verified_at": datetime.now().astimezone().isoformat(), "passed": True,
            "research_direction": manifest["research_direction"],
            "papers": len(papers), "physical_pages": sum(p["pages"] for p in papers.values()),
            "questions": len(questions), "categories": dict(Counter(q["category"] for q in questions)),
            "languages": dict(Counter(q["language"] for q in questions)),
            "category_languages": {category: dict(Counter(q["language"] for q in questions
                                                          if q["category"] == category))
                                   for category in sorted(CATEGORIES)},
            "paper_question_coverage": {pid: sum(pid in q["paper_ids"] for q in questions) for pid in papers},
            "evidence_locations": len(evidence), "anchor_checks": anchor_checks,
            "paper_checks": paper_checks, "corpus_chunks": len(corpus),
            "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
            "manifest_sha256": payload["manifest_sha256"],
            "corpus_path": str(CORPUS.relative_to(ROOT)) if CORPUS.is_relative_to(ROOT) else str(CORPUS),
            "corpus_sha256": hashlib.sha256(serialized).hexdigest(),
            "libraries": {name: version(name) for name in ("PyMuPDF", "langchain-text-splitters")},
            "boundary": "核验覆盖结构、原文哈希和锚点定位；参考答案已按原文编写，未经独立人工复核。没有运行模型评分或系统性能评测；锚点所在块只作定位，不等同于完整chunk相关性标注。"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="显式下载尚未准备的固定论文并校验哈希")
    parser.add_argument("--output", type=Path, required=True, help="不存在的新核验报告路径")
    parser.add_argument("--corpus", type=Path, default=CORPUS, help="新版本语料输出路径，不覆盖旧实验语料")
    args = parser.parse_args()
    CORPUS = args.corpus.resolve()
    if args.output.exists():
        raise FileExistsError("核验报告已存在，请使用新的输出路径")
    result = prepare_and_verify(args.download)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "papers", "questions", "categories", "languages", "corpus_chunks")}, ensure_ascii=False))
