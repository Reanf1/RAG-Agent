"""本地 M3E 与 Chroma：持久化、增量入库、Top-K 检索及按文档删除。"""

import os
import json
from hashlib import sha256
from functools import lru_cache
from threading import Lock
from pathlib import Path
from time import perf_counter

import numpy as np

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import chroma_metadata, load_config
from src.utils.chroma_path import chroma_persist_path


_embedding_load_lock = Lock()


def get_embeddings():
    """首次加载互斥；等待者复用缓存，构造失败后允许下次重试。"""
    with _embedding_load_lock:
        return _load_embeddings()


@lru_cache(maxsize=1)
def _load_embeddings() -> HuggingFaceEmbeddings:
    """首次调用才加载唯一配置的本地模型；加载失败直接报错，不转云端。"""
    project_root = Path(__file__).resolve().parents[2]
    config = load_config()["embedding"]
    model_path = project_root / config["local_path"]
    if not model_path.is_dir():
        raise FileNotFoundError(f"本地 Embedding 模型不存在：{model_path}，请先按用户手册下载权重")
    # 缓存留在项目模型目录；不在页面启动或模型编码时自动下载权重。
    os.environ.setdefault("HF_HOME", str(project_root / "data/models/.hf-runtime"))
    return HuggingFaceEmbeddings(
        model_name=str(model_path),
        model_kwargs={"device": config["device"], "local_files_only": True, "trust_remote_code": False},
        encode_kwargs={"normalize_embeddings": True, "batch_size": config["batch_size"]},
    )


get_embeddings.cache_clear = _load_embeddings.cache_clear


class VectorStore:
    """沿用参考项目的 LangChain Chroma；仅维护一种本地存储。"""

    def __init__(self, persist_directory: str | Path | None = None,
                 embeddings: Embeddings | None = None):
        """默认读取 YAML；测试可注入小型 Embedding，数据库仍使用真实 Chroma。"""
        config = load_config()
        retrieval = config["retrieval"]
        if retrieval["vector_store"] != "chroma":
            raise ValueError("当前业务仅实现选定的 Chroma，FAISS 仅用于对比实验")
        project_root = Path(__file__).resolve().parents[2]
        directory = Path(persist_directory or config["paths"]["vector_index"])
        if not directory.is_absolute():
            directory = project_root / directory
        self.directory = directory.resolve()
        self.top_k = retrieval["top_k"]
        expected = chroma_metadata(config)
        self._store = Chroma(
            collection_name=retrieval["collection_name"],
            embedding_function=embeddings,
            persist_directory=chroma_persist_path(self.directory),
            client_settings=Settings(is_persistent=True, anonymized_telemetry=False),
            collection_metadata=expected,
        )
        # 已存在集合不会被 get_or_create 覆盖；避免不同模型或索引参数静默混用。
        actual = self._store._collection.metadata or {}
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("已有索引的模型版本或索引参数与配置不一致，请使用新索引目录重建")
    def _ensure_embeddings(self):
        """只有向量写入/查询才加载模型，BM25 读取正文无需准备权重。"""
        if self._store.embeddings is None:
            self._store._embedding_function = get_embeddings()

    def count(self) -> int:
        """返回真实块数；数据库异常直接上报，不把错误伪装成空库。"""
        return self._store._collection.count()

    def add_chunks(self, chunks: list[Document]) -> int:
        """按稳定chunk_id去重，仅编码新块；返回实际新增数量。

        先合并本批重复ID，再逐批查询数据库中已存在的ID，最后仅写缺失块。
        因此重复上传或失败重试不会重新编码已成功写入的块。写入不是整批
        文档级事务：后续批次失败时保留之前批次，重试从缺失块继续。
        """
        unique = {}
        for chunk in chunks:
            for key in ("chunk_id", "doc_id"):
                if not isinstance(chunk.metadata.get(key), str) or not chunk.metadata[key].strip():
                    raise ValueError(f"文档块必须包含非空 {key}，请先使用加载器和分块模块")
            unique[chunk.metadata["chunk_id"]] = chunk
        documents = list(unique.values())
        added = 0
        # 分批查重/写入，避免大批文献超过数据库单次写入限制；失败可重复提交。
        for start in range(0, len(documents), 500):
            batch = documents[start:start + 500]
            ids = [chunk.metadata["chunk_id"] for chunk in batch]
            existing = set(self._store.get(ids=ids, include=[])["ids"])
            new = [chunk for chunk in batch if chunk.metadata["chunk_id"] not in existing]
            if new:
                self._ensure_embeddings()
                self._store.add_documents(new, ids=[chunk.metadata["chunk_id"] for chunk in new])
                added += len(new)
        return added

    def list_chunks(self, doc_id: str | None = None) -> list[Document]:
        """读取正文和完整来源；可作为后续 BM25 的语料，不计算 Embedding。"""
        result = self._store.get(where={"doc_id": doc_id} if doc_id is not None else None,
                                 include=["documents", "metadatas"])
        return [Document(page_content=text, metadata=metadata)
                for text, metadata in zip(result["documents"], result["metadatas"])]

    def search(self, query: str, k: int | None = None,
               doc_id: str | None = None) -> list[tuple[Document, float]]:
        """按余弦相似度降序返回 Top-K 正文、来源与分数。

        k 默认取 YAML；不足 k 个时返回全部匹配块。Chroma 返回的
        余弦距离越小越相关，转换为相似度 1−距离；保留负分，不设阈值。
        """
        k = self.top_k if k is None else k
        if type(k) is not int or k <= 0:
            raise ValueError("k 必须为正整数")
        # 空问题无需读取数据库或计算查询向量。
        if not query.strip():
            return []
        where = {"doc_id": doc_id} if doc_id is not None else None
        if where:
            # 限定一篇论文采用已有向量的精确余弦排名；索引损坏时由上层明确报错。
            stored = self._store.get(where=where, include=["documents", "metadatas", "embeddings"])
            if not stored["ids"]:
                return []
            self._ensure_embeddings()
            vectors = np.asarray(stored["embeddings"], dtype=float)
            query_vector = np.asarray(self._store.embeddings.embed_query(query), dtype=float)
            norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
            if np.any(norms == 0):
                raise ValueError("持久化向量或查询向量为零，无法计算余弦相似度")
            scores = np.clip(vectors @ query_vector / norms, -1.0, 1.0)
            ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], stored["ids"][i]))[:k]
            return [(Document(page_content=stored["documents"][i], metadata=stored["metadatas"][i]),
                     float(scores[i])) for i in ranked]
        count = self.count()
        if count == 0:
            return []
        self._ensure_embeddings()
        results = self._store.similarity_search_with_score(query, k=min(k, count))
        return [(document, 1 - distance) for document, distance in results]

    def delete_document(self, doc_id: str) -> int:
        """删除指定文档的正文、向量与来源，返回删除块数；保留原始文件。"""
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("doc_id 必须为非空字符串")
        where = {"doc_id": doc_id}
        count = len(self._store.get(where=where, include=[])["ids"])
        if count:
            self._store._collection.delete(where=where)
        return count


def _write_expected_count(task, count):
    """把本次预期块数写到原文目录的轻量标记，用于页面判断"部分入库"。"""
    path = Path(task["path"]).parent / ".index_count"
    path.write_text(str(count), encoding="utf-8")
    return count


def batch_build_index(tasks: list[dict], raw_dir: str | Path,
                      max_file_size_mb: int = 20, retry_failed: bool = False,
                      vector_store: VectorStore | None = None):
    """逐文件加载、分块、批量向量化并持久化，更新任务并产出进度。

    原始加载接口仍独立使用。索引失败保留原文/加载结果与已写入块；
    重试只补缺失块，每个目标任务最多尝试一次，不重建已有索引。
    """
    from src.chunking import split_documents
    from src.data_loader import batch_import, document_lock

    statuses = {"failed"} if retry_failed else {"pending", "loading", "chunking", "indexing"}
    targets = [task for task in tasks if task["status"] in statuses
               or (not retry_failed and task["status"] == "success" and not task.get("indexed", False))]
    total = len(targets)
    yield {"completed": 0, "total": total}
    for index, task in enumerate(targets):
        start = perf_counter()
        needs_loading = not task["path"]
        task.update(status="loading" if needs_loading else "chunking", error="",
                    indexed=False, chunk_count=0, processed_chunks=0, added_chunks=0)
        try:
            with document_lock(raw_dir, sha256(task["data"]).hexdigest()):
                # 已删除的原文不能被旧任务内存中的 Document 重新入库。
                if not needs_loading and not Path(task["path"]).is_file():
                    raise FileNotFoundError("原文已删除或缺失，请恢复后重新导入")
                if needs_loading:
                    # 复用原有格式/大小校验和安全保存，不向页面展示短暂的加载成功。
                    for _ in batch_import([task], raw_dir, max_file_size_mb):
                        if task["status"] == "loading":
                            yield {"completed": index, "total": total}
                else:
                    task["attempts"] += 1
                if task["status"] != "failed":
                    task["status"] = "chunking"
                    yield {"completed": index, "total": total}
                    chunks = split_documents(task["documents"])
                    if not chunks:
                        raise ValueError("文档没有可索引的非空正文")
                    task.update(status="indexing", chunk_count=len(chunks))
                    yield {"completed": index, "total": total}
                    # 第一个有效文档才初始化本地模型/数据库，一批复用同一实例。
                    if vector_store is None:
                        vector_store = VectorStore()
                    expected_ids = {chunk.metadata["chunk_id"] for chunk in chunks}
                    existing_ids = {chunk.metadata["chunk_id"]
                                    for chunk in vector_store.list_chunks(chunks[0].metadata["doc_id"])}
                    if existing_ids - expected_ids:
                        # 参数或解析结果变化时保留旧库，不能把另一套块混入同一文档。
                        raise ValueError("已有文档的分块或解析结果与本次不一致，请在新索引目录重建后启用；原索引未修改")
                    # 只记本次预期块数，供页面区分"已向量化/部分入库"；不记路径与ID清单。
                    _write_expected_count(task, len(chunks))
                    for offset in range(0, len(chunks), 500):
                        batch = chunks[offset:offset + 500]
                        task["added_chunks"] += vector_store.add_chunks(batch)
                        task["processed_chunks"] = offset + len(batch)
                        task["index_total"] = vector_store.count()
                        yield {"completed": index, "total": total}
                    actual_ids = {chunk.metadata["chunk_id"]
                                  for chunk in vector_store.list_chunks(chunks[0].metadata["doc_id"])}
                    if actual_ids != expected_ids:
                        raise ValueError("实际索引块与预期集合不一致，本次入库未完成")
                    task.update(status="success", indexed=True)
        except Exception as error:
            # 失败只影响当前文档；保留已落盘块，供下一次查重恢复。
            task.update(status="failed", error=f"{type(error).__name__}: {error}")
        task["elapsed"] = round(perf_counter() - start, 3)
        yield {"completed": index + 1, "total": total}
