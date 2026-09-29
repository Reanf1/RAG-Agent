"""本地 M3E 与 Chroma：持久化、增量入库、Top-K 检索及按文档删除。"""

import os
from functools import lru_cache
from pathlib import Path

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import load_config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
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
        self.top_k = retrieval["top_k"]
        expected = {"embedding_model": config["embedding"]["model"],
                    "embedding_revision": config["embedding"]["revision"],
                    "hnsw:space": "cosine",
                    "hnsw:search_ef": retrieval["search_ef"],
                    "hnsw:num_threads": retrieval["index_threads"]}
        self._store = Chroma(
            collection_name=retrieval["collection_name"],
            embedding_function=embeddings if embeddings is not None else get_embeddings(),
            persist_directory=str(directory.resolve()),
            client_settings=Settings(is_persistent=True, anonymized_telemetry=False),
            collection_metadata=expected,
        )
        # 已存在集合不会被 get_or_create 覆盖；避免不同模型或索引参数静默混用。
        actual = self._store._collection.metadata or {}
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("已有索引的模型版本或索引参数与配置不一致，请使用新索引目录重建")

    def count(self) -> int:
        """返回真实块数；数据库异常直接上报，不把错误伪装成空库。"""
        return self._store._collection.count()

    def add_chunks(self, chunks: list[Document]) -> int:
        """按稳定 chunk_id 去重，仅编码新块；返回实际新增数量。"""
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
        """返回 Top-K 与余弦相似度（1−距离），分数不是命中概率。"""
        k = self.top_k if k is None else k
        if type(k) is not int or k <= 0:
            raise ValueError("k 必须为正整数")
        where = {"doc_id": doc_id} if doc_id is not None else None
        count = len(self._store.get(where=where, include=[])["ids"]) if where else self.count()
        if not query.strip() or count == 0:
            return []
        results = self._store.similarity_search_with_score(query, k=min(k, count), filter=where)
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
