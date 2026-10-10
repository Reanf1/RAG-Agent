"""在同一固定检索样本上实测两种 Embedding；论文问题与原文语言一致。"""

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import statistics
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/raw/embedding_benchmark"
MODEL_DIR = PROJECT_ROOT / "data/models"
MODELS = [
    ("BAAI/bge-large-zh-v1.5", "79e7739b6ab944e86d6171e44d24c997fc1e0116",
     "pytorch_model.bin", "为这个句子生成表示以用于检索相关文章："),
    ("moka-ai/m3e-base", "764b537a0e50e5c7d64db883f2d2e051cbe3c64c", "model.safetensors", ""),
]
DATASETS = [
    ("C-MTEB/T2Retrieval", "8731a845f1bf500a4f111cf1070785c793d10e64",
     "data/corpus-00000-of-00001-8afe7b7a7eca49e3.parquet"),
    ("C-MTEB/T2Retrieval", "8731a845f1bf500a4f111cf1070785c793d10e64",
     "data/queries-00000-of-00001-930bf3b805a80dd9.parquet"),
    ("C-MTEB/T2Retrieval-qrels", "1c83b8d1544e529875e3f6930f3a1fcf749a8e97",
     "data/dev-00000-of-00001-92ed0416056ff7e1.parquet"),
]
SEED = 20260929


def download_inputs(include_datasets=True):
    """显式 --download 时准备公开权重和数据；默认运行只读本地文件。"""
    from huggingface_hub import hf_hub_download, snapshot_download

    if include_datasets:
        for repo, revision, filename in DATASETS:
            hf_hub_download(repo, filename, repo_type="dataset", revision=revision, local_dir=DATA_DIR)
    for name, revision, weight, _ in MODELS:
        snapshot_download(name, revision=revision, local_dir=MODEL_DIR / name.split("/")[-1],
                          allow_patterns=["*.json", "vocab.txt", weight], max_workers=2)


def prepare_sample():
    """固定种子抽取 50 个问题，保留全部标注相关段落，再加入 500 个候选。"""
    import pyarrow.parquet as pq

    corpus_path, queries_path, qrels_path = [DATA_DIR / item[2] for item in DATASETS]
    queries = pq.read_table(queries_path).to_pylist()
    qrels = {}
    for row in pq.read_table(qrels_path).to_pylist():
        if row["score"] > 0:
            qrels.setdefault(row["qid"], set()).add(row["pid"])
    rng = random.Random(SEED)
    queries = rng.sample([query for query in queries if query["id"] in qrels], 50)
    positives = set().union(*(qrels[query["id"]] for query in queries))
    parquet = pq.ParquetFile(corpus_path)
    extra_rows = set(rng.sample(range(parquet.metadata.num_rows), 500))
    corpus = []
    offset = 0
    for batch in parquet.iter_batches(batch_size=10000):
        for index, row in enumerate(batch.to_pylist(), offset):
            if index in extra_rows or row["id"] in positives:
                corpus.append(row)
        offset += batch.num_rows
    assert positives <= {row["id"] for row in corpus}, "不能漏掉标注相关段落"
    sample = {"seed": SEED, "corpus": corpus, "queries": [
        {**query, "relevant_ids": sorted(qrels[query["id"]])} for query in queries
    ]}
    payload = json.dumps(sample, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    (DATA_DIR / "sample.json").write_bytes(payload)
    return sample, hashlib.sha256(payload).hexdigest()


def evaluate_rankings(scores, sample):
    """计算 Hit@5、Recall@5、MRR@10 与逐题排名，保留可复核的原始结果。"""
    import numpy as np

    details = []
    for query, values in zip(sample["queries"], scores):
        order = np.argsort(-values, kind="stable")[:10]
        ids = [sample["corpus"][index]["id"] for index in order]
        relevant = set(query["relevant_ids"])
        first_rank = next((rank for rank, pid in enumerate(ids, 1) if pid in relevant), None)
        details.append({
            "query_id": query["id"], "relevant_ids": sorted(relevant), "top10_ids": ids,
            "top10_scores": [float(values[index]) for index in order],
            "hit_at_5": bool(relevant.intersection(ids[:5])),
            "recall_at_5": len(relevant.intersection(ids[:5])) / len(relevant),
            "reciprocal_rank_at_10": 1 / first_rank if first_rank else 0,
            **({"group": query["group"]} if "group" in query else {}),
        })
    result = {"hit_at_5": statistics.mean(row["hit_at_5"] for row in details),
            "recall_at_5": statistics.mean(row["recall_at_5"] for row in details),
            "mrr_at_10": statistics.mean(row["reciprocal_rank_at_10"] for row in details),
            "queries": details}
    if "group" in sample["queries"][0]:
        result["groups"] = {}
        for group in sorted({row["group"] for row in details}):
            rows = [row for row in details if row["group"] == group]
            result["groups"][group] = {"query_count": len(rows),
                "hit_at_5": statistics.mean(row["hit_at_5"] for row in rows),
                "recall_at_5": statistics.mean(row["recall_at_5"] for row in rows),
                "mrr_at_10": statistics.mean(row["reciprocal_rank_at_10"] for row in rows)}
        result["macro"] = {key: statistics.mean(group[key] for group in result["groups"].values())
                           for key in ("hit_at_5", "recall_at_5", "mrr_at_10")}
        result["worst_group_hit_at_5"] = min(group["hit_at_5"] for group in result["groups"].values())
    return result


def benchmark_model(model_info, sample, runs=3):
    """CPU、FP32、4 线程、批次 32；正常三轮、数值复核一轮，下载不计时。"""
    import numpy as np
    from langchain_huggingface import HuggingFaceEmbeddings

    name, revision, weight, instruction = model_info
    path = MODEL_DIR / name.split("/")[-1]
    started = perf_counter()
    embeddings = HuggingFaceEmbeddings(
        model_name=str(path),
        model_kwargs={"device": "cpu", "local_files_only": True, "trust_remote_code": False},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    embeddings._client.max_seq_length = 512
    load_seconds = perf_counter() - started
    texts = [row["text"] for row in sample["corpus"]]
    queries = [instruction + row["text"] for row in sample["queries"]]
    # 这里只统计未截断长度，不把超长 token 序列送入模型；实际编码按 512 截断。
    token_lengths = [len(ids) for ids in embeddings._client.tokenizer(
        texts, truncation=False, verbose=False)["input_ids"]]
    embeddings.embed_documents(texts[:8])
    embeddings.embed_query(queries[0])
    document_times, query_times = [], []
    # 分别计量两种正文语言，仍然使用完整混合候选库计算排名。
    languages = sorted({row.get("language", "all") for row in sample["corpus"]})
    language_indices = {language: [i for i, row in enumerate(sample["corpus"])
                                  if row.get("language", "all") == language] for language in languages}
    language_times = {language: [] for language in languages}
    for run in range(runs):
        started = perf_counter()
        document_vectors = None
        for language, indices in language_indices.items():
            batch_started = perf_counter()
            vectors = np.asarray(embeddings.embed_documents([texts[i] for i in indices]))
            language_times[language].append(perf_counter() - batch_started)
            if document_vectors is None:
                document_vectors = np.empty((len(texts), vectors.shape[1]))
            document_vectors[indices] = vectors
        document_times.append(perf_counter() - started)
        query_vectors = []
        for query in queries:
            started = perf_counter()
            query_vectors.append(embeddings.embed_query(query))
            query_times.append(perf_counter() - started)
        print(name, "完成第", run + 1, "轮，文档编码秒", round(document_times[-1], 3), flush=True)
    query_vectors = np.asarray(query_vectors)
    assert np.isfinite(document_vectors).all() and np.isfinite(query_vectors).all(), "向量包含非有限值"
    document_norms = np.einsum("pd,pd->p", document_vectors, document_vectors)
    query_norms = np.einsum("qd,qd->q", query_vectors, query_vectors)
    norm_error = max(float(np.max(np.abs(document_norms - 1))), float(np.max(np.abs(query_norms - 1))))
    assert norm_error < 1e-5, "编码向量未正确归一化"
    # 当前 macOS 的矩阵乘法产生过数值警告；显式点积避免依赖该 BLAS 路径。
    scores = np.einsum("qd,pd->qp", query_vectors, document_vectors, optimize=False)
    assert np.isfinite(scores).all(), "相似度包含非有限值"
    result = evaluate_rankings(scores, sample)
    result.update({
        "model": name, "revision": revision, "weight_bytes": (path / weight).stat().st_size,
        "dimension": int(document_vectors.shape[1]), "query_instruction": instruction,
        "load_seconds": load_seconds, "document_seconds_runs": document_times,
        "document_seconds_median": statistics.median(document_times),
        "documents_per_second": len(texts) / statistics.median(document_times),
        "query_seconds_runs": query_times, "query_ms_median": statistics.median(query_times) * 1000,
        "query_ms_p95": float(np.percentile(query_times, 95)) * 1000,
        "truncated_document_count": sum(length > 512 for length in token_lengths),
        "vectors_and_scores_finite": True, "max_squared_norm_error": norm_error,
        "document_languages": {language: {
            "document_count": len(language_indices[language]), "seconds_runs": times,
            "seconds_median": statistics.median(times),
            "documents_per_second": len(language_indices[language]) / statistics.median(times),
        } for language, times in language_times.items()},
    })
    if "groups" in result:
        for group, values in result["groups"].items():
            times = [elapsed for i, elapsed in enumerate(query_times)
                     if sample["queries"][i % len(queries)]["group"] == group]
            values["query_ms_median"] = statistics.median(times) * 1000
            values["query_ms_p95"] = float(np.percentile(times, 95)) * 1000
    del embeddings
    gc.collect()
    return result


def main():
    """默认离线复现实验；输出独立 JSON，不修改课程最终评测集。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="先下载固定版本的公开模型和数据")
    parser.add_argument("--output", type=Path, help="结果路径；论文基准与旧中文实验默认分开保存")
    parser.add_argument("--verify-results", type=Path, help="仅重编码一轮，复核指定结果的指标与排名")
    parser.add_argument("--sample", type=Path, help="使用已准备、已锁定的论文双语样本 JSON")
    args = parser.parse_args()
    if args.output is None:
        filename = "Embedding论文双语对比结果.json" if args.sample else "Embedding对比结果.json"
        args.output = PROJECT_ROOT / "reports/5_1_3 向量化与存储" / filename
    os.environ.setdefault("HF_HOME", str(MODEL_DIR / ".hf-runtime"))
    os.environ["HF_HUB_OFFLINE"] = "0" if args.download else "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.download:
        download_inputs(include_datasets=not args.sample)
    import torch
    torch.set_num_threads(4)
    if args.sample:
        payload = args.sample.read_bytes()
        sample, sample_hash = json.loads(payload), hashlib.sha256(payload).hexdigest()
    else:
        sample, sample_hash = prepare_sample()
    runs = 1 if args.verify_results else 3
    expected = json.loads(args.verify_results.read_text(encoding="utf-8")) if args.verify_results else None
    if expected:
        assert sample_hash == expected["sample_sha256"], "复核样本必须与原实验一致"
        assert args.output.resolve() != args.verify_results.resolve(), "复核结果必须另存，不能覆盖原始计时"
    result = {"dataset_revisions": [list(item) for item in DATASETS], "sample_sha256": sample_hash,
              "seed": SEED, "query_count": len(sample["queries"]), "document_count": len(sample["corpus"]),
              "python": sys.version.split()[0], "system": platform.platform(), "device": "cpu",
              "dtype": "float32", "cpu_threads": 4, "batch_size": 32, "max_tokens": 512, "runs": runs,
              "models": []}
    if args.sample:
        result.pop("dataset_revisions")
        result.pop("seed")
        result.update({key: sample[key] for key in ("name", "manifest_sha256", "papers", "chunking", "selection_rule")})
    result["started_at"] = datetime.now(timezone.utc).isoformat()
    result["package_versions"] = {name: version(name) for name in (
        "torch", "transformers", "sentence-transformers", "langchain-huggingface", "numpy", "PyMuPDF")}
    print("固定样本", result["query_count"], "个问题", result["document_count"], "个候选", sample_hash, flush=True)
    for model in MODELS:
        model_result = benchmark_model(model, sample, runs)
        if expected:
            original = next(row for row in expected["models"] if row["model"] == model[0])
            for key in ("hit_at_5", "recall_at_5", "mrr_at_10"):
                assert abs(model_result[key] - original[key]) < 1e-10, f"复核指标发生变化：{model[0]} {key}"
            model_result["top10_order_matches_original"] = all(
                current["top10_ids"] == old["top10_ids"]
                for current, old in zip(model_result["queries"], original["queries"])
            )
            model_result["metrics_match_original"] = True
        result["models"].append(model_result)
        # 每个模型完成后保存，保留实际已完成结果，不把未运行模型写入表格。
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for model in result["models"]:
        print(model["model"], "Hit@5", model["hit_at_5"], "MRR@10", model["mrr_at_10"],
              "段/秒", model["documents_per_second"], "查询毫秒", model["query_ms_median"], flush=True)


if __name__ == "__main__":
    main()
