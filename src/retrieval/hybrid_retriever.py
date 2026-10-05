"""手写 RRF：按排名融合向量与 BM25，稳定块 ID 用于合并来源。"""

from langchain_core.documents import Document

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import Reranker, is_image_placeholder
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config


def rrf_fusion(vector_results: list[tuple[Document, float]],
               bm25_results: list[tuple[Document, float]],
               rrf_k: int = 60) -> list[tuple[Document, float]]:
    """返回全部融合候选，分数为各召回分支的1/(rrf_k+排名)之和。

    向量相似度与BM25分数的尺度不同，不能直接相加；RRF只取从1开始的
    排名。同一chunk_id在单个分支只贡献一次，在两个分支命中则累加。
    rrf_k越大，名次之间的差距越平缓；此参数与最终返回文档数k无关。
    """
    if type(rrf_k) is not int or rrf_k < 0:
        raise ValueError("rrf_k 必须为非负整数")
    scores, documents = {}, {}
    for results in (vector_results, bm25_results):
        seen = set()
        for rank, (document, _) in enumerate(results, 1):
            chunk_id = document.metadata.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("RRF 候选必须包含非空 chunk_id")
            # 每路同一块只计首次出现；两路同时命中则累加各自的排名贡献。
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            documents.setdefault(chunk_id, document)
    # 同分按稳定块 ID 排序，重复查询和独立公式核验可得到一致结果。
    ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [(documents[chunk_id], scores[chunk_id]) for chunk_id in ranked]


class HybridRetriever:
    """共享一个 Chroma，两路融合后可选择模型精排，不生成答案。"""

    def __init__(self, vector_store: VectorStore | None = None):
        config = load_config()["retrieval"]
        self.candidate_k = config["candidate_k"]
        self.rrf_k = config["rrf_k"]
        if type(self.candidate_k) is not int or self.candidate_k <= 0:
            raise ValueError("candidate_k 必须为正整数")
        if type(self.rrf_k) is not int or self.rrf_k < 0:
            raise ValueError("rrf_k 必须为非负整数")
        self.vector_store = vector_store if vector_store is not None else VectorStore()
        self.top_k = self.vector_store.top_k

    def search(self, query: str, k: int | None = None,
               doc_id: str | None = None, *, rerank: bool = False) -> list[tuple[Document, float]]:
        """两路先融合；启用重排时对候选精排后才截取最终 Top-K。"""
        k = self.top_k if k is None else k
        if type(k) is not int or k <= 0:
            raise ValueError("k 必须为正整数")
        if not query.strip():
            return []
        candidate_k = max(self.candidate_k, k)
        vector_results = self.vector_store.search(query, k=candidate_k, doc_id=doc_id)
        # 每次读取当前正文，新增/删除后无需维护第二套语料或缓存失效规则。
        bm25_results = BM25Retriever(self.vector_store).search(query, k=candidate_k, doc_id=doc_id)
        fused = rrf_fusion(vector_results, bm25_results, self.rrf_k)
        # 加载时保留图片位置供原文查看，检索时不让占位文字挤占Top-20正文候选。
        fused = [(document, score) for document, score in fused if not is_image_placeholder(document.page_content)]
        if rerank:
            return Reranker().rerank(query, fused[:candidate_k], k=k)
        return fused[:k]
