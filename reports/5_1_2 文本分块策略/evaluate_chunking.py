"""固定论文、题目和模型，比较五组分块与返回K；不覆盖用户索引或旧实验。"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PROFILES = [("fixed256", "fixed", 256), ("fixed512", "fixed", 512),
            ("fixed1024", "fixed", 1024), ("recursive", "recursive", None),
            ("semantic", "semantic", None)]


def digest(path):
    """冻结输入资料，禁止依据输出修改金标准。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(ranked, gold, k):
    """相关性按论文指纹和物理页计算，同页重复块不增加召回。"""
    found, first = set(), None
    for rank, item in enumerate(ranked[:k], 1):
        meta = item["metadata"]
        pages = {(meta["doc_id"], p) for p in range(meta["page_number"], meta.get("page_end", meta["page_number"]) + 1)}
        matched = pages & gold
        found |= matched
        if matched and first is None:
            first = rank
    return {"hit": bool(found), "mrr": 1 / first if first else 0,
            "recall": len(found) / len(gold)}


def summarize(rows):
    """保留全部题目的分母，另外报告中英和题型分组。"""
    return {"questions": len(rows), "latency_mean_ms": statistics.mean(r["seconds"] * 1000 for r in rows),
            "at_k": {str(k): {key: statistics.mean(r["at_k"][str(k)][key] for r in rows)
                               for key in ("hit", "mrr", "recall")} for k in (3, 5, 10)},
            "by_language": {lang: {key: statistics.mean(r["at_k"]["5"][key] for r in rows if r["language"] == lang)
                                    for key in ("hit", "mrr", "recall")} for lang in ("zh", "en")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="读取完整断点，保留已完成题目与索引")
    args = parser.parse_args()
    if not args.resume and (args.root.exists() or args.output.exists()):
        raise FileExistsError("请使用新的实验目录和结果文件")
    args.root.mkdir(parents=True, exist_ok=args.resume)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import torch
    torch.set_num_threads(4)
    import src.utils.config as config_module
    config = config_module.load_config()
    original = deepcopy(config)
    config_module.load_config = lambda: deepcopy(config)
    from src.data_loader.pdf_loader import load_pdf
    from src.chunking import split_documents
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.hybrid_retriever import HybridRetriever
    dataset = ROOT / "reports/评测集.json"
    manifest_path = ROOT / "reports/5_5_1 评测集构建/论文清单.json"
    questions = json.loads(dataset.read_text())
    papers = json.loads(manifest_path.read_text())["papers"]
    ids = {p["id"]: p["doc_id"] for p in papers}
    documents = []
    for paper in papers:
        path = ROOT / paper["local_path"]
        assert digest(path) == paper["doc_id"]
        documents.extend(load_pdf(path))
    report = {"status": "running", "started_at": datetime.now().astimezone().isoformat(),
              "dataset_sha256": digest(dataset), "manifest_sha256": digest(manifest_path),
              "config": original, "environment": {"python": platform.python_version(), "platform": platform.platform()},
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src").rglob("*.py"))},
              "method": "同一12篇PDF、60题、M3E、RRF候选20+BGE；页级标注；K共用Top20排名，耗时是Top20检索", 
              "profiles": {}, "rows": []}
    if args.resume:
        saved = json.loads(args.output.read_text())
        if saved["dataset_sha256"] != report["dataset_sha256"] or saved["config"] != original:
            raise ValueError("断点输入或业务配置已改变，不能混入旧实验")
        report = saved
        report.setdefault("resumes", []).append(datetime.now().astimezone().isoformat())
    def save():
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)  # 写完再替换，避免中断产生半份JSON。
    save()
    order = random.Random(20261004).sample(questions, len(questions))
    for name, strategy, size in PROFILES:
        completed = {r["id"] for r in report["rows"] if r["profile"] == name}
        if len(completed) == len(questions) and name in report["profiles"]:
            continue
        config["paths"]["vector_index"] = str(args.root / name)
        chunks = split_documents(documents, strategy=strategy, chunk_size=size)
        store = VectorStore()
        started = perf_counter()
        if store.count():
            assert {d.metadata["chunk_id"] for d in store.list_chunks()} == {d.metadata["chunk_id"] for d in chunks}
            build_seconds = report["profiles"].get(name, {}).get("build_seconds")
        else:
            assert store.add_chunks(chunks) == len(chunks)
            build_seconds = perf_counter() - started
        retriever = HybridRetriever(store)
        # 首次模型加载和预热单列；查询均获取同一Top20，再截取各K计算指标。
        started = perf_counter()
        retriever.search(order[0]["question"], k=20, rerank=True)
        warmup_seconds = perf_counter() - started
        for index, question in enumerate(order, 1):
            if question["id"] in completed:
                continue
            started = perf_counter()
            results = retriever.search(question["question"], k=20, rerank=True)
            elapsed = perf_counter() - started
            keys = {"doc_id", "page", "page_number", "page_end", "source_file", "chunk_id", "chunk_strategy",
                    "chunk_size", "chunk_overlap", "start_index", "end_index", "content_type", "block_type", "table_id"}
            ranked = [{"metadata": {key: value for key, value in d.metadata.items() if key in keys},
                       "text": d.page_content, "score": score} for d, score in results]
            gold = {(ids[e["paper_id"]], e["page_number"]) for e in question["evidence"]}
            report["rows"].append({"id": question["id"], "profile": name, "category": question["category"],
                                   "language": question["language"], "seconds": elapsed, "top20": ranked,
                                   "at_k": {str(k): metrics(ranked, gold, k) for k in (3, 5, 10)}})
            save()
            print(f"{name} {index}/60 {question['id']} {elapsed:.3f}s", flush=True)
        report["profiles"][name] = {"strategy": strategy, "chunk_size": chunks[0].metadata["chunk_size"],
            "overlap": original["chunking"]["chunk_overlap"], "chunks": len(chunks), "build_seconds": build_seconds,
            "warmup_seconds": warmup_seconds, **summarize([r for r in report["rows"] if r["profile"] == name])}
        save()
    report.update(status="completed", completed_at=datetime.now().astimezone().isoformat())
    save()
    print(json.dumps(report["profiles"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
