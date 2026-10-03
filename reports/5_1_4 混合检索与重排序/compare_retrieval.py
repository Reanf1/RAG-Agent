"""在锁定的中英论文基准上实测向量、RRF、RRF+BGE 的 Top-5 质量。"""

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import tempfile
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))

METRICS = ("hit_at_5", "recall_at_5", "mrr_at_5")
METHODS = ("vector", "hybrid", "hybrid_rerank")
LABELS = {"vector": "纯向量", "hybrid": "混合检索", "hybrid_rerank": "混合+重排序"}


def evaluate_ranking(ranked_ids: list[str], relevant_ids: list[str]) -> dict:
    """按标注块 ID 计算；第六名不计入 Top-5，未命中时 MRR@5 为 0。"""
    relevant = set(relevant_ids)
    if not relevant or len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("相关标注不能为空，检索结果不能包含重复块 ID")
    top5 = ranked_ids[:5]
    hits = len(relevant.intersection(top5))
    first = next((rank for rank, chunk_id in enumerate(top5, 1) if chunk_id in relevant), None)
    return {"hit_at_5": bool(hits), "recall_at_5": hits / len(relevant),
            "mrr_at_5": 1 / first if first else 0.0,
            "first_relevant_rank": first, "relevant_returned": hits}


def summarize(rows: list[dict]) -> dict:
    """总体按查询平均；语言组等权宏平均单独列出，不能重复计配对意图。"""
    groups = {}
    for group in sorted({row["group"] for row in rows}):
        selected = [row for row in rows if row["group"] == group]
        groups[group] = {"query_count": len(selected),
                         **{key: statistics.mean(row[key] for row in selected) for key in METRICS}}
    elapsed = [ms for row in rows for ms in row["latency_ms_runs"]]
    import numpy as np
    return {"query_count": len(rows), "hit_count": sum(row["hit_at_5"] for row in rows),
            **{key: statistics.mean(row[key] for row in rows) for key in METRICS},
            "groups": groups,
            "macro": {key: statistics.mean(group[key] for group in groups.values()) for key in METRICS},
            "worst_group_hit_at_5": min(group["hit_at_5"] for group in groups.values()),
            "latency_mean_ms": statistics.mean(elapsed), "latency_median_ms": statistics.median(elapsed),
            "latency_p95_ms": float(np.percentile(elapsed, 95))}


def validate_sample(sample: dict):
    """标注与实际语料必须对应；错误数据应停止，不在评测时自动修改答案。"""
    ids = [row["id"] for row in sample["corpus"]]
    query_ids = [row["id"] for row in sample["queries"]]
    if not ids or not query_ids or len(set(ids)) != len(ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("语料/问题不能为空或包含重复 ID")
    for query in sample["queries"]:
        if not query["relevant_ids"] or not set(query["relevant_ids"]) <= set(ids):
            raise ValueError(f"问题 {query['id']} 的相关标注缺失或不在语料中")


def main():
    """仅运行当前配置，不改题、不调参；重复轮次用于计时与排名稳定性核验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="预热后重复轮次，默认 3")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/5_1_4 混合检索与重排序/检索三档对比结果.json")
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("runs 必须为正整数")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data/models/.hf-runtime"))
    import torch
    from langchain_core.documents import Document
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import get_reranker
    from src.utils.config import load_config

    torch.set_num_threads(4)
    config = load_config()
    if config["retrieval"]["top_k"] != 5 or config["retrieval"]["candidate_k"] != 20:
        raise ValueError("本实验固定最终 Top-5 和候选 Top-20，请先核对配置")
    sample_path = PROJECT_ROOT / "data/raw/embedding_papers/sample.json"
    manifest_path = PROJECT_ROOT / "reports/5_1_3 向量化与存储/Embedding论文双语评测集.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    validate_sample(sample)
    assert sample["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "标注版本变化"
    documents = [Document(page_content=row["text"], metadata={
        "chunk_id": row["id"], "doc_id": row["paper_id"], "source_file": row["paper_id"] + ".pdf",
        **{key: row[key] for key in ("page_number", "language", "start_index", "end_index", "content_type")}})
        for row in sample["corpus"]]
    results = {method: {} for method in METHODS}
    candidate_checks = {}
    round_summaries = []
    with tempfile.TemporaryDirectory(prefix="rag-quality-") as directory:
        store = VectorStore(directory)
        started = perf_counter()
        assert store.add_chunks(documents) == len(documents)
        build_seconds = perf_counter() - started
        print(f"真实 M3E 编码/索引完成：{len(documents)} 块，{build_seconds:.2f} 秒", flush=True)
        retriever = HybridRetriever(store)
        calls = {"vector": lambda text: store.search(text, k=5),
                 "hybrid": lambda text: retriever.search(text, k=5),
                 "hybrid_rerank": lambda text: retriever.search(text, k=5, rerank=True)}
        # 模型加载及预热单列，不能混进测量轮次的查询延迟。
        started = perf_counter()
        for call in calls.values():
            call(sample["queries"][0]["text"])
        warmup_seconds = perf_counter() - started
        for run in range(args.runs):
            queries = random.Random(20260929 + run).sample(sample["queries"], len(sample["queries"]))
            current = {method: [] for method in METHODS}
            for index, query in enumerate(queries):
                # 轮换方法顺序，避免每次都把同一方法放在最冷/最热的位置。
                offset = (run + index) % len(METHODS)
                for method in METHODS[offset:] + METHODS[:offset]:
                    started = perf_counter()
                    found = calls[method](query["text"])
                    elapsed = (perf_counter() - started) * 1000
                    ids = [doc.metadata["chunk_id"] for doc, _ in found]
                    metrics = evaluate_ranking(ids, query["relevant_ids"])
                    row = results[method].get(query["id"])
                    if row is None:
                        row = {"query_id": query["id"], "query": query["text"], "pair_id": query["pair_id"],
                               "group": query["group"], "paper_id": query["paper_id"],
                               "relevant_ids": query["relevant_ids"], **metrics,
                               "top5": [{"chunk_id": doc.metadata["chunk_id"], "score": score,
                                         "source_file": doc.metadata["source_file"], "page_number": doc.metadata["page_number"]}
                                        for doc, score in found], "latency_ms_runs": [], "ranking_changes": []}
                        results[method][query["id"]] = row
                    elif ids != [item["chunk_id"] for item in row["top5"]]:
                        row["ranking_changes"].append({"run": run + 1, "ids": ids, **metrics})
                    row["latency_ms_runs"].append(elapsed)
                    current[method].append({**metrics, "group": query["group"], "latency_ms_runs": [elapsed]})
                if run == 0:
                    pool = retriever.search(query["text"], k=20)
                    pool_ids = [doc.metadata["chunk_id"] for doc, _ in pool]
                    assert pool_ids[:5] == [item["chunk_id"] for item in results["hybrid"][query["id"]]["top5"]]
                    assert set(item["chunk_id"] for item in results["hybrid_rerank"][query["id"]]["top5"]) <= set(pool_ids)
                    candidate_checks[query["id"]] = {"candidate_ids": pool_ids,
                        "candidate_hit_at_20": bool(set(pool_ids).intersection(query["relevant_ids"]))}
                if (index + 1) % 16 == 0:
                    print(f"第 {run + 1}/{args.runs} 轮：{index + 1}/{len(queries)} 题完成", flush=True)
            round_summaries.append({method: summarize(current[method]) for method in METHODS})
    methods = {}
    for method in METHODS:
        rows = [results[method][query["id"]] for query in sample["queries"]]
        methods[method] = {"label": LABELS[method], **summarize(rows), "queries": rows}
    report = {"evaluated_at": datetime.now().astimezone().isoformat(), "offline": True,
              "environment": {"platform": platform.platform(), "python": platform.python_version(),
                              "torch_threads": torch.get_num_threads(),
                              "libraries": {name: version(name) for name in
                                  ("torch", "sentence-transformers", "transformers", "chromadb", "rank-bm25")}},
              "embedding": config["embedding"], "retrieval": config["retrieval"], "chunking": sample["chunking"],
              "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
              "manifest_sha256": sample["manifest_sha256"], "papers": len(sample["papers"]),
              "corpus_chunks": len(documents), "query_count": len(sample["queries"]),
              "intent_count": len({row["pair_id"] for row in sample["queries"]}),
              "annotation_note": json.loads(manifest_path.read_text())["annotation_note"],
              "runs": args.runs, "metrics_round": 1, "seed": 20260929, "build_seconds": build_seconds,
              "warmup_seconds": warmup_seconds, "methods": methods, "round_summaries": round_summaries,
              "candidate_checks": candidate_checks,
              "reranker_instance_reused": get_reranker() is get_reranker()}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({method: {key: methods[method][key] for key in
        ("hit_at_5", "recall_at_5", "mrr_at_5", "latency_mean_ms")} for method in METHODS}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
