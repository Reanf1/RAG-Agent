"""同一批本地 M3E 向量比较 Chroma 与 FAISS；FAISS 只用于实验。"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from importlib import import_module
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
WORK_DIR = PROJECT_ROOT / "data/raw/vector_store_benchmark"


def write_json(path, value):
    """保留真实结果与正文/来源；FAISS 的正文存储也计入磁盘占用。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def open_chroma(directory, search_ef):
    """只调整候选搜索范围；显式提供向量，不启用默认 Embedding 或遥测。"""
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(directory), settings=Settings(
        anonymized_telemetry=False)).get_or_create_collection(
            "paper_chunks", embedding_function=None,
            metadata={"hnsw:space": "cosine", "hnsw:num_threads": 4,
                      "hnsw:search_ef": search_ef})


def records_from_sample(sample):
    """实验用 paper_id 标识文档；业务使用加载器生成的文件 SHA 文档 ID。"""
    return [{"id": row["id"], "text": row["text"], "metadata": {
        "chunk_id": row["id"], "doc_id": row["paper_id"],
        "source_file": row["paper_id"] + ".pdf",
        **{key: row[key] for key in ("paper_id", "language", "page_number",
                                    "start_index", "end_index", "content_type")},
    }} for row in sample["corpus"]]


def search_index(backend, index, queries, records):
    """两库逐题查询 Top-10；计时含 SDK 和结果映射，不含 Embedding。"""
    rankings, timings = [], []
    for vector in queries:
        started = perf_counter()
        if backend == "chroma":
            result = index.query(query_embeddings=[vector.tolist()], n_results=10,
                                 include=["distances"])
            ids = result["ids"][0]
            scores = [1 - distance for distance in result["distances"][0]]
        else:
            distances, integers = index.search(vector.reshape(1, -1), 10)
            ids = [records[int(i)]["id"] for i in integers[0]]
            scores = distances[0].tolist()
        timings.append((perf_counter() - started) * 1000)
        rankings.append({"ids": ids, "scores": scores})
    return rankings, timings


def verify_in_new_process(backend, directory, vectors_path, search_ef):
    """在独立 Python 进程重开索引，验证全部正文/来源及查询排名。"""
    import faiss
    import numpy as np

    faiss.omp_set_num_threads(4)
    records = json.loads((directory / "records.json").read_text())
    queries = np.load(vectors_path)["queries"]
    started = perf_counter()
    if backend == "chroma":
        index = open_chroma(directory, search_ef)
        loaded = index.get(include=["documents", "metadatas"])
        actual = {key: (text, metadata) for key, text, metadata in zip(
            loaded["ids"], loaded["documents"], loaded["metadatas"])}
        assert actual == {row["id"]: (row["text"], row["metadata"]) for row in records}
        count = index.count()
    else:
        index = faiss.read_index(str(directory / "index.faiss"))
        count = index.ntotal
        assert sorted(faiss.vector_to_array(index.id_map).tolist()) == list(range(len(records)))
    load_ms = (perf_counter() - started) * 1000
    rankings, _ = search_index(backend, index, queries, records)
    return {"count": count, "load_and_read_ms": load_ms, "rankings": rankings,
            "records_sha256": hashlib.sha256((directory / "records.json").read_bytes()).hexdigest()}


def benchmark(backend, directory, documents, queries, records, search_ef):
    """738 块初始建库、83 块增量写入；三轮查询后删除/恢复一篇论文。"""
    import faiss
    import numpy as np

    directory.mkdir(parents=True)
    start = perf_counter()
    index = open_chroma(directory, search_ef) if backend == "chroma" else faiss.IndexIDMap2(
        faiss.IndexFlatIP(documents.shape[1]))
    init_ms = (perf_counter() - start) * 1000

    def add(indices):
        if backend == "chroma":
            index.add(ids=[records[i]["id"] for i in indices],
                      embeddings=documents[indices].tolist(),
                      documents=[records[i]["text"] for i in indices],
                      metadatas=[records[i]["metadata"] for i in indices])
        else:
            index.add_with_ids(documents[indices], np.asarray(indices, dtype="int64"))

    start = perf_counter()
    add(list(range(len(records) - 83)))
    initial_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    add(list(range(len(records) - 83, len(records))))
    incremental_ms = (perf_counter() - start) * 1000
    # 预热一次，随后三个完整查询轮次；各轮保持相同的逐题顺序。
    search_index(backend, index, queries[:1], records)
    timings = []
    for _ in range(3):
        rankings, measured = search_index(backend, index, queries, records)
        timings.extend(measured)
    doc_id = records[0]["metadata"]["doc_id"]
    deleted = [i for i, row in enumerate(records) if row["metadata"]["doc_id"] == doc_id]
    start = perf_counter()
    if backend == "chroma":
        index.delete(where={"doc_id": doc_id})
        remaining = index.count()
    else:
        index.remove_ids(np.asarray(deleted, dtype="int64"))
        remaining = index.ntotal
    delete_ms = (perf_counter() - start) * 1000
    assert remaining == len(records) - len(deleted)
    add(deleted)
    restored_rankings, _ = search_index(backend, index, queries, records)
    start = perf_counter()
    if backend == "faiss":
        faiss.write_index(index, str(directory / "index.faiss"))
    write_json(directory / "records.json", records)
    save_ms = (perf_counter() - start) * 1000
    # Chroma records.json 仅是验证用副本，不纳入数据库磁盘大小。
    size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()
               and not (backend == "chroma" and path.name == "records.json"))
    output = subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
        "--verify-backend", backend, "--directory", str(directory),
        "--vectors", str(WORK_DIR / "vectors.npz"), "--search-ef", str(search_ef)], text=True)
    reopened = json.loads(output)
    assert reopened["count"] == len(records)
    assert reopened["records_sha256"] == hashlib.sha256((directory / "records.json").read_bytes()).hexdigest()
    assert [row["ids"] for row in reopened["rankings"]] == [row["ids"] for row in restored_rankings]
    return {"init_ms": init_ms, "initial_add_ms": initial_ms,
            "incremental_add_ms": incremental_ms, "delete_ms": delete_ms,
            "deleted_count": len(deleted), "count_after_delete": remaining,
            "explicit_save_ms": save_ms, "disk_bytes": size,
            "query_ms": timings, "query_p50_ms": float(np.percentile(timings, 50)),
            "query_p95_ms": float(np.percentile(timings, 95)),
            "restart_load_and_read_ms": reopened["load_and_read_ms"],
            "restart_verified": True, "rankings": rankings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-backend", choices=["chroma", "faiss"])
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--vectors", type=Path)
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--reuse-vectors", action="store_true")
    parser.add_argument("--search-ef", type=int, default=100)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/5_1_3 向量化与存储/向量数据库对比结果.json")
    args = parser.parse_args()
    if args.verify_backend:
        print(json.dumps(verify_in_new_process(args.verify_backend, args.directory, args.vectors, args.search_ef)))
        return
    import numpy as np
    from src.utils.config import load_config
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = PROJECT_ROOT / "data/raw/embedding_papers/sample.json"
    payload = sample_path.read_bytes()
    sample = json.loads(payload)
    records = records_from_sample(sample)
    if args.encode_only:
        # 本机 FAISS/PyTorch 同进程编码会段错误，准备向量的进程不导入 FAISS。
        import torch
        from src.retrieval.vector_store import get_embeddings

        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        print(f"正在从本地 M3E 编码 {len(records)} 个分块与 {len(sample['queries'])} 条同语言查询……", flush=True)
        started = perf_counter()
        embeddings = get_embeddings()
        documents = np.asarray(embeddings.embed_documents([row["text"] for row in records]), dtype="float32")
        queries = np.asarray(embeddings.embed_documents([row["text"] for row in sample["queries"]]), dtype="float32")
        np.savez(WORK_DIR / "vectors.npz", documents=documents, queries=queries)
        write_json(WORK_DIR / "encoding.json", {"seconds": perf_counter() - started,
            "sample_sha256": hashlib.sha256(payload).hexdigest(), "embedding": load_config()["embedding"]})
        return
    if not args.reuse_vectors:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--encode-only"], check=True)
    encoding = json.loads((WORK_DIR / "encoding.json").read_text())
    assert encoding["sample_sha256"] == hashlib.sha256(payload).hexdigest()
    assert encoding["embedding"] == load_config()["embedding"]
    import faiss
    evaluate_rankings = import_module("reports.5_1_3 向量化与存储.compare_embeddings").evaluate_rankings

    faiss.omp_set_num_threads(4)
    arrays = np.load(WORK_DIR / "vectors.npz")
    documents, queries = arrays["documents"], arrays["queries"]
    encode_seconds = encoding["seconds"]
    assert documents.shape == (len(records), 768) and queries.shape == (len(sample["queries"]), 768)
    assert np.isfinite(documents).all() and np.isfinite(queries).all()
    assert np.allclose(np.linalg.norm(documents, axis=1), 1, atol=1e-5)
    assert np.allclose(np.linalg.norm(queries, axis=1), 1, atol=1e-5)
    result = {"created_at": datetime.now(timezone.utc).isoformat(),
        "sample_sha256": hashlib.sha256(payload).hexdigest(),
        "vectors_sha256": hashlib.sha256((WORK_DIR / "vectors.npz").read_bytes()).hexdigest(),
        "embedding": load_config()["embedding"], "encoding_seconds_excluded": encode_seconds,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": platform.processor(), "threads": 4,
            "versions": {name: version(name) for name in (
                "numpy", "chromadb", "langchain-chroma", "faiss-cpu", "posthog", "torch")}},
        "selection_rule": {"minimum_top5_overlap": 0.99, "maximum_query_p95_ms": 20},
        "chroma_hnsw": {"search_ef": args.search_ef, "construction_ef": 100, "M": 16,
                        "space": "cosine", "num_threads": 4, "batch_size": 100, "sync_threshold": 1000},
        "corpus_count": len(records), "query_count": len(queries), "runs": {}}
    prefix = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    for run in range(3):
        # 交替执行次序，避免始终让一种后端先运行。
        for backend in (["chroma", "faiss"] if run % 2 == 0 else ["faiss", "chroma"]):
            measured = benchmark(backend, WORK_DIR / f"{prefix}-{backend}-{run}", documents, queries, records, args.search_ef)
            result["runs"].setdefault(backend, []).append(measured)
            print(f"{backend} 第 {run + 1} 轮：P95={measured['query_p95_ms']:.3f} ms，重启验证通过", flush=True)
    for backend, runs in result["runs"].items():
        # 用返回排名构造分数表，复用已有评测公式；未返回项不计入 Top-10。
        scores = np.full((len(queries), len(records)), -np.inf)
        id_to_index = {row["id"]: i for i, row in enumerate(records)}
        for q, ranking in enumerate(runs[0]["rankings"]):
            for rank, chunk_id in enumerate(ranking["ids"]):
                scores[q, id_to_index[chunk_id]] = 10 - rank
        result.setdefault("quality", {})[backend] = evaluate_rankings(scores, sample)
        for row, ranking in zip(result["quality"][backend]["queries"], runs[0]["rankings"]):
            row["top10_scores"] = ranking["scores"]
    exact = result["runs"]["faiss"][0]["rankings"]
    overlaps = [len(set(row["ids"][:5]) & set(reference["ids"][:5])) / 5
                for run in result["runs"]["chroma"] for row, reference in zip(run["rankings"], exact)]
    result["top5_overlap_with_exact"] = statistics.mean(overlaps)
    result["selected"] = "chroma" if result["top5_overlap_with_exact"] >= 0.99 and max(
        run["query_p95_ms"] for run in result["runs"]["chroma"]) < 20 else "faiss"
    write_json(args.output, result)
    print(f"实验完成，规则选择：{result['selected']}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    main()
