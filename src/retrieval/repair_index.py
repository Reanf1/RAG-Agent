"""停机维护：由已有真实正文重建新索引，独立进程校验后才替换，原索引保留为备份。"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import numpy as np

from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config


def content_digest(chunks):
    """核对所有块ID、正文与来源元数据，不以块数相等代替内容相等。"""
    records = sorted((doc.metadata["chunk_id"], doc.page_content, doc.metadata) for doc in chunks)
    return sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def rebuild_index(source, output, *, embeddings=None):
    """只读取原库的正文；损坏的HNSW向量不参与重建，也不向原库写入／删除块。"""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not (source / "chroma.sqlite3").is_file():
        raise FileNotFoundError("原索引不存在，不能用空库冒充修复")
    if output.exists():
        raise FileExistsError("新索引目录已存在，请选择新目录；不覆盖失败记录或原索引")
    original = VectorStore(source, embeddings)
    chunks = original.list_chunks()
    if not chunks:
        raise ValueError("原索引没有正文，不能执行重建")
    digest = content_digest(chunks)
    target = VectorStore(output, embeddings)
    if target.add_chunks(chunks) != len(chunks):
        raise ValueError("新索引写入块数不完整")
    manifest = {"source": str(source), "output": str(output), "chunks": len(chunks),
                "documents": len({doc.metadata["doc_id"] for doc in chunks}),
                "content_sha256": digest, "verified_after_restart": False}
    (output / "repair.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_index(output, *, embeddings=None):
    """在重建进程退出后执行：逐批读全部持久化向量，并验证真实HNSW查询。"""
    output = Path(output).resolve()
    manifest = json.loads((output / "repair.json").read_text(encoding="utf-8"))
    store = VectorStore(output, embeddings)
    chunks = store.list_chunks()
    if len(chunks) != manifest["chunks"] or content_digest(chunks) != manifest["content_sha256"]:
        raise ValueError("新索引的块ID、正文或来源与原库不一致")
    first = None
    for start in range(0, len(chunks), 500):
        ids = [doc.metadata["chunk_id"] for doc in chunks[start:start + 500]]
        # 直接读取持久化向量；此处不能调用带临时恢复的search来掩盖Label错误。
        stored = store._store.get(ids=ids, include=["embeddings"])
        vectors = np.asarray(stored["embeddings"], dtype=float)
        if (set(stored["ids"]) != set(ids) or vectors.ndim != 2 or len(vectors) != len(ids)
                or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0)):
            raise ValueError("新索引存在缺失、非有限或零向量")
        if first is None:
            first = vectors[0].tolist()
    # 使用已有向量做全库原生查询，不额外编码、不依赖查询临时恢复。
    result = store._store._collection.query(query_embeddings=[first], n_results=1, include=["distances"])
    if (not result["ids"][0] or not np.isfinite(result["distances"][0]).all()
            or abs(result["distances"][0][0]) > 1e-4):
        raise ValueError("新索引未通过自身向量查询校验")
    # 停机期间原库若被其他请求改写，不能用过时快照替换它。
    original = VectorStore(manifest["source"], embeddings)
    if content_digest(original.list_chunks()) != manifest["content_sha256"]:
        raise ValueError("原索引在重建期间发生变化，拒绝替换，请停止应用后重试")
    manifest["verified_after_restart"] = True
    (output / "repair.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def activate_index(source, output):
    """两个数据库进程都退出后同目录替换；替换失败回滚，不删除原始索引。"""
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = json.loads((output / "repair.json").read_text(encoding="utf-8"))
    if source == output or source.parent != output.parent or manifest["source"] != str(source):
        raise ValueError("替换要求原库与新库为同一父目录下的不同目录，且来源相同")
    if not manifest["verified_after_restart"]:
        raise ValueError("新库尚未通过独立进程重启校验，不能替换")
    backup = source.with_name(source.name + "-backup-" + uuid4().hex[:12])
    source.rename(backup)
    try:
        output.rename(source)
    except OSError:
        backup.rename(source)
        raise
    return str(backup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="默认读取config.yaml中的索引目录")
    parser.add_argument("--output", type=Path, help="默认在原库旁新建唯一目录；已有目录拒绝覆盖")
    parser.add_argument("--activate", action="store_true", help="停止Streamlit后使用；校验完成才替换并保留备份")
    parser.add_argument("--worker", choices=("build", "verify"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = args.source or Path(load_config()["paths"]["vector_index"])
    source = (source if source.is_absolute() else root / source).resolve()
    output = args.output or source.with_name(source.name + "-rebuilt-" + uuid4().hex[:12])
    output = (output if output.is_absolute() else root / output).resolve()
    if args.worker:
        result = rebuild_index(source, output) if args.worker == "build" else verify_index(output)
    else:
        # Windows文件句柄只能在子进程退出后释放，不能在Chroma仍运行时移动索引目录。
        for phase in ("build", "verify"):
            subprocess.run([sys.executable, "-m", "src.retrieval.repair_index", "--worker", phase,
                            "--source", str(source), "--output", str(output)], cwd=root, check=True)
        result = json.loads((output / "repair.json").read_text(encoding="utf-8"))
        if args.activate:
            result["backup"] = activate_index(source, output)
            result["active_index"] = str(source)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
