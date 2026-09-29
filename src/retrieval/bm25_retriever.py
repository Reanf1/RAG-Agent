"""从 Chroma 正文构建轻量 BM25，沿用上游的中英分词与 BM25Okapi。"""

import re
import unicodedata

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.retrieval.vector_store import VectorStore


def tokenize(text: str) -> list[str]:
    """英文/数字按词并统一小写，中文按单字；文档与问题使用相同规则。"""
    # 学术 PDF 中常有全角英文/数字，只规范检索词项，保留 Document 原文。
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"[a-zA-Z0-9]+", normalized) + re.findall(r"[\u4e00-\u9fff]", normalized)


class BM25Retriever:
    """小规模知识库的内存索引；正文/来源仍以持久化 Chroma 为准。"""

    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store if vector_store is not None else VectorStore()
        self.top_k = self.vector_store.top_k
        self.rebuild()

    def rebuild(self) -> int:
        """重新读取文档块；新增/删除后调用，不重新向量化，返回可检索块数。"""
        documents, corpus = [], []
        for document in self.vector_store.list_chunks():
            tokens = tokenize(document.page_content)
            # 纯空白或标点无法检索；全空词表会导致 BM25Okapi 计算异常。
            if tokens:
                documents.append(document)
                corpus.append(tokens)
        index = BM25Okapi(corpus) if corpus else None
        self._documents = documents
        self._terms = [set(tokens) for tokens in corpus]
        self._index = index
        return len(documents)

    def search(self, query: str, k: int | None = None,
               doc_id: str | None = None) -> list[tuple[Document, float]]:
        """按 BM25 原始分数降序返回有关键词交集的 Top-K，保留正文/来源。"""
        k = self.top_k if k is None else k
        if type(k) is not int or k <= 0:
            raise ValueError("k 必须为正整数")
        tokens = tokenize(query)
        if not tokens or self._index is None:
            return []
        query_terms = set(tokens)
        candidates = [i for i, document in enumerate(self._documents)
                      if (doc_id is None or document.metadata["doc_id"] == doc_id)
                      and query_terms.intersection(self._terms[i])]
        if not candidates:
            return []
        scores = self._index.get_scores(tokens)
        # 单篇/小语料的 IDF 可为零或负数，不能用 score > 0 判断是否命中。
        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)[:k]
        return [(self._documents[i], float(scores[i])) for i in ranked]
