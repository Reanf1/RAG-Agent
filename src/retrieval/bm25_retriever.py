"""从 Chroma 正文构建轻量 BM25，沿用上游的中英分词与 BM25Okapi。"""

import re
import unicodedata
from functools import lru_cache
from threading import Lock

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.retrieval.vector_store import VectorStore


# 双语论文常用词项映射；只扩展检索词，不生成或修改论文事实。
ACADEMIC_TERMS = {"位置编码": "positional position encoding embeddings", "位置嵌入": "positional position embeddings",
                  "自注意力": "self attention", "注意力": "attention", "编码器": "encoder",
                  "解码器": "decoder", "图像块": "image patches", "数据集": "dataset datasets",
                  "训练": "training", "预训练": "pretraining pretrained", "准确率": "accuracy",
                  "损失函数": "loss", "对比学习": "contrastive learning", "自监督": "self supervised"}


def expand_academic_query(query: str) -> str:
    """保留中文原问题，附加实际命中的英文术语，供BM25和BGE共同理解。"""
    terms = list(dict.fromkeys(term for chinese, english in ACADEMIC_TERMS.items() if chinese in query
                               for term in english.split()))
    return query + ("\n" + " ".join(terms) if terms else "")


def tokenize(text: str) -> list[str]:
    """英文/数字按词并统一小写，中文按单字；文档与问题使用相同规则。"""
    # 学术 PDF 中常有全角英文/数字，只规范检索词项，保留 Document 原文。
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"[a-zA-Z0-9]+", normalized) + re.findall(r"[\u4e00-\u9fff]", normalized)


_corpus_lock = Lock()


@lru_cache(maxsize=4)
def _build_corpus(texts: tuple[str, ...]):
    """仅复用最多四份正文相同的分词／评分索引，来源始终读取当前数据库。"""
    positions, corpus = [], []
    for index, text in enumerate(texts):
        tokens = tokenize(text)
        if tokens:
            positions.append(index)
            corpus.append(tokens)
    return positions, [frozenset(tokens) for tokens in corpus], BM25Okapi(corpus) if corpus else None


class BM25Retriever:
    """小规模知识库的内存索引；正文/来源仍以持久化 Chroma 为准。"""

    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store if vector_store is not None else VectorStore()
        self.top_k = self.vector_store.top_k
        self.rebuild()

    def rebuild(self) -> int:
        """重新读取文档块；新增/删除后调用，不重新向量化，返回可检索块数。"""
        documents = self.vector_store.list_chunks()
        # 用完整有序正文作缓存键，不以块数/时间戳猜版本；同数量替换也会重建。
        with _corpus_lock:
            positions, terms, index = _build_corpus(tuple(document.page_content for document in documents))
        self._documents = [documents[position] for position in positions]
        self._terms, self._index = terms, index
        return len(self._documents)

    def search(self, query: str, k: int | None = None,
               doc_id: str | None = None) -> list[tuple[Document, float]]:
        """按 BM25 原始分数降序返回有关键词交集的 Top-K，保留正文/来源。"""
        k = self.top_k if k is None else k
        if type(k) is not int or k <= 0:
            raise ValueError("k 必须为正整数")
        tokens = tokenize(expand_academic_query(query))
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
