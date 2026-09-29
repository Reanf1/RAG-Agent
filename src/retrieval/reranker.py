"""本地 BGE Cross-Encoder：成对评分，保留原文和来源后精排。"""

import math
import os
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from src.utils.config import load_config


@lru_cache(maxsize=1)
def get_reranker():
    """首次有候选时才加载本地模型；失败直接报错，不下载或切换方案。"""
    config = load_config()["retrieval"]
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / config["reranker_local_path"]
    if not model_path.is_dir():
        raise FileNotFoundError(f"本地重排模型不存在：{model_path}，请先按用户手册下载权重")
    os.environ.setdefault("HF_HOME", str(project_root / "data/models/.hf-runtime"))
    # 在设定可写缓存后才导入模型库，普通检索和页面启动无需初始化它。
    from sentence_transformers import CrossEncoder
    from torch.nn import Sigmoid
    return CrossEncoder(
        str(model_path), device=config["reranker_device"],
        max_length=config["reranker_max_length"],
        local_files_only=True, trust_remote_code=False,
        default_activation_function=Sigmoid(),
    )


class Reranker:
    """仅精排传入候选；返回模型相关性分数，不与 RRF 分数相加。"""

    def __init__(self):
        config = load_config()["retrieval"]
        self.top_k = config["top_k"]
        self.batch_size = config["reranker_batch_size"]
        for key in ("reranker_batch_size", "reranker_max_length"):
            if type(config[key]) is not int or config[key] <= 0:
                raise ValueError(f"{key} 必须为正整数")

    def rerank(self, query: str, candidates: list[tuple[Document, float]],
               k: int | None = None) -> list[tuple[Document, float]]:
        """按模型分数降序返回 Top-K；同分保持候选原顺序，不修改 Document。"""
        k = self.top_k if k is None else k
        if type(k) is not int or k <= 0:
            raise ValueError("k 必须为正整数")
        if not query.strip() or not candidates:
            return []
        # 与参考项目相同：每个问题—正文对输出一个分数，sigmoid 对应 normalize=True。
        pairs = [[query, document.page_content] for document, _ in candidates]
        scores = get_reranker().predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        if len(scores) != len(candidates):
            raise ValueError("重排模型返回的分数数量与候选数量不一致")
        scored = [(document, float(score)) for (document, _), score in zip(candidates, scores)]
        if not all(math.isfinite(score) for _, score in scored):
            raise ValueError("重排模型返回非有限分数")
        # sigmoid 只将分数映射到 0~1，不代表已经校准的命中概率。
        return sorted(scored, key=lambda item: item[1], reverse=True)[:k]
