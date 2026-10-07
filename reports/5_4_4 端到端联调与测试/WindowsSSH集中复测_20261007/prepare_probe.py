"""Windows原生功能复测准备：复制真实资料和索引，仅修改副本的来源路径。"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project, target = args.project.resolve(), args.root.resolve()
    if target.exists() or args.output.exists():
        raise FileExistsError("不覆盖已有测试目录或证据")
    sys.path.insert(0, str(project))
    import numpy as np
    from src.retrieval.repair_index import content_digest
    from src.retrieval.vector_store import VectorStore
    from src.utils.config import load_config

    config = load_config()
    raw = (project / config["paths"]["raw_documents"]).resolve()
    index = (project / config["paths"]["vector_index"]).resolve()
    original = VectorStore(index)
    chunks = original.list_chunks()
    digest = content_digest(chunks)
    target.mkdir(parents=True)
    shutil.copytree(raw, target / "raw")
    shutil.copytree(index, target / "index")
    copied = VectorStore(target / "index")
    if content_digest(copied.list_chunks()) != digest:
        raise ValueError("复制后的正文和来源与原索引不一致")
    # 块、正文及向量不变；原文预览和副本管理统一指向独立测试目录。
    ids, metadata = [], []
    for chunk in chunks:
        source = Path(chunk.metadata["source"]).resolve()
        relative = source.relative_to(raw)
        ids.append(chunk.metadata["chunk_id"])
        metadata.append({**chunk.metadata, "source": str(target / "raw" / relative)})
    copied._store._collection.update(ids=ids, metadatas=metadata)
    for manifest in (target / "raw").glob("*/.index_status.json"):
        state = json.loads(manifest.read_text(encoding="utf-8"))
        if state["index_directory"] == str(index):
            state["index_directory"] = str(target / "index")
            manifest.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    stored = copied._store.get(include=["embeddings"])
    vectors = np.asarray(stored["embeddings"], dtype=float)
    if (len(stored["ids"]) != len(chunks) or vectors.ndim != 2
            or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0)):
        raise ValueError("持久化向量存在缺失、非有限值或零向量")
    result = copied._store._collection.query(query_embeddings=[vectors[0].tolist()], n_results=1,
                                             include=["distances"])
    if not result["ids"][0] or abs(result["distances"][0][0]) > 1e-4:
        raise ValueError("原生HNSW自身向量查询失败")
    unchanged = content_digest(original.list_chunks()) == digest
    if not unchanged:
        raise ValueError("原索引在准备期间改变，拒绝继续")
    files = [{"file": path.name, "doc_id": path.parent.name, "bytes": path.stat().st_size,
              "sha256": sha256(path.read_bytes()).hexdigest()}
             for path in sorted(raw.glob("*/*")) if path.is_file() and path.suffix.lower() in {".pdf", ".md", ".txt", ".docx"}]
    report = {"scope": __doc__, "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip(),
              "source_raw": str(raw), "source_index": str(index), "probe_root": str(target),
              "chunks": len(chunks), "documents": len({chunk.metadata["doc_id"] for chunk in chunks}),
              "source_content_sha256": digest, "original_unchanged": unchanged, "files": files,
              "vectors_shape": list(vectors.shape), "native_hnsw_self_distance": result["distances"][0][0],
              "passed": True}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
