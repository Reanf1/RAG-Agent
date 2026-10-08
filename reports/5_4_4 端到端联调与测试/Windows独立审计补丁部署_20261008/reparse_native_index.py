"""停机部署专用：从原文建新库，独立进程验证后切换；保留旧库和旧旁注。"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys


def save(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "verify", "activate"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project))
    from src.utils.config import load_config
    config = load_config()
    expected = json.loads((args.run / "original-fingerprint.json").read_text(encoding="utf-8"))
    active = (args.project / config["paths"]["vector_index"]).resolve()
    fresh = active.with_name(active.name + "-reparsed-20261008")
    backup = active.with_name(active.name + "-before-audit-20261008")
    output = args.run / ("reparse-" + args.phase + ".json")
    if output.exists():
        raise FileExistsError("不覆盖已有重建证据")
    for row in expected["files"]:
        source = Path(expected["source_raw"]) / row["doc_id"] / row["file"]
        assert sha256(source.read_bytes()).hexdigest() == row["sha256"], "原文指纹变化，停止部署"

    if args.phase == "activate":
        verified = json.loads((args.run / "reparse-verify.json").read_text(encoding="utf-8"))
        assert verified["passed"] and fresh.is_dir() and not backup.exists()
        built = json.loads((args.run / "reparse-build.json").read_text(encoding="utf-8"))
        # 此进程不打开 Chroma，避免 Windows 文件句柄阻止目录替换。
        sidecars = args.run / "original-index-status"
        sidecars.mkdir()
        for row in built["documents"]:
            source = Path(expected["source_raw"]) / row["doc_id"] / ".index_status.json"
            if source.exists():
                shutil.copy2(source, sidecars / (row["doc_id"] + ".json"))
        active.rename(backup)
        try:
            fresh.rename(active)
        except OSError:
            backup.rename(active)
            raise
        # 只更新索引旁注；PDF/Word/文本原件不写入，旧旁注已备份。
        for row in built["documents"]:
            source = Path(expected["source_raw"]) / row["doc_id"] / ".index_status.json"
            state = {"index_directory": str(active), "collection": config["retrieval"]["collection_name"],
                     "expected_ids": row["chunk_ids"], "complete": True}
            temporary = source.with_suffix(".audit.tmp")
            temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            temporary.replace(source)
        result = {"active_index": str(active), "backup_index": str(backup), "passed": True}
    else:
        import numpy as np
        from src.retrieval.repair_index import content_digest
        from src.retrieval.vector_store import VectorStore
        if args.phase == "build":
            from src.chunking import split_documents
            from src.data_loader import load_document
            if fresh.exists():
                raise FileExistsError("新库目录已存在，拒绝覆盖")
            store = VectorStore(fresh)
            documents = []
            for row in expected["files"]:
                source = Path(expected["source_raw"]) / row["doc_id"] / row["file"]
                chunks = split_documents(load_document(source))
                assert chunks and store.add_chunks(chunks) == len(chunks)
                documents.append({**row, "chunk_ids": [chunk.metadata["chunk_id"] for chunk in chunks]})
            result = {"new_index": str(fresh), "documents": documents, "chunks": store.count(),
                      "content_sha256": content_digest(store.list_chunks()), "passed": True}
        else:
            built = json.loads((args.run / "reparse-build.json").read_text(encoding="utf-8"))
            store = VectorStore(fresh)
            chunks = store.list_chunks()
            assert len(chunks) == built["chunks"] and content_digest(chunks) == built["content_sha256"]
            stored = store._store.get(include=["embeddings"])
            vectors = np.asarray(stored["embeddings"], dtype=float)
            assert len(stored["ids"]) == len(chunks) and vectors.ndim == 2
            assert np.isfinite(vectors).all() and np.all(np.linalg.norm(vectors, axis=1) > 0)
            query = store._store._collection.query(query_embeddings=[vectors[0].tolist()], n_results=1, include=["distances"])
            assert query["ids"][0] and abs(query["distances"][0][0]) < 1e-4
            old = VectorStore(active)
            assert content_digest(old.list_chunks()) == expected["source_content_sha256"]
            result = {"chunks": len(chunks), "documents": len({chunk.metadata["doc_id"] for chunk in chunks}),
                      "vectors_shape": list(vectors.shape), "content_sha256": content_digest(chunks),
                      "old_content_unchanged": True, "native_self_distance": query["distances"][0][0], "passed": True}
    save(output, result)
    print(json.dumps({"phase": args.phase, "passed": result["passed"], "chunks": result.get("chunks")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
