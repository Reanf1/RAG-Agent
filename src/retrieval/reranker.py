"""本地 BGE Cross-Encoder：成对评分，保留原文和来源后精排。"""

import math
import os
import re
from copy import deepcopy
from functools import lru_cache
from threading import Lock
from pathlib import Path

from langchain_core.documents import Document

from src.utils.config import load_config


def is_image_placeholder(text: str) -> bool:
    """只有图像占位描述与序号的块不能成为正文证据；混有正文/公式的块仍保留。"""
    if "[图像区域" not in text:
        return False
    body = re.sub(r"\[图像区域[^\]\n]*\]", "", text)
    # 只剔除剩余内容全是序号、空白的占位块，不误删公式符号和纯数字表格。
    return not body.strip() or bool(re.fullmatch(r"[\s\d]+", body))


_reranker_load_lock = Lock()


def get_reranker():
    """首次加载互斥；等待者复用缓存，构造失败后允许下次重试。"""
    with _reranker_load_lock:
        return _load_reranker()


@lru_cache(maxsize=1)
def _load_reranker():
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


get_reranker.cache_clear = _load_reranker.cache_clear


class Reranker:
    """仅精排传入候选；返回模型相关性分数，不与 RRF 分数相加。"""

    def __init__(self):
        config = load_config()["retrieval"]
        self.top_k = config["top_k"]
        self.batch_size = config["reranker_batch_size"]
        self.max_length = config["reranker_max_length"]
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
        model = get_reranker()
        passages = [self._passages(query, document, model.tokenizer) for document, _ in candidates]
        pairs = [[query, text] for windows in passages for text in windows]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        if len(scores) != len(pairs):
            raise ValueError("重排模型返回的分数数量与候选数量不一致")
        scored, offset = [], 0
        for (document, _), windows in zip(candidates, passages):
            window_scores = [float(score) for score in scores[offset:offset + len(windows)]]
            if not all(math.isfinite(score) for score in window_scores):
                raise ValueError("重排模型返回非有限分数")
            best = max(range(len(windows)), key=lambda index: window_scores[index])
            if len(windows) > 1:
                document = deepcopy(document)
                # 仅作本次检索摘录；原索引、块ID和表格行来源保持不变。
                document.metadata["rerank_excerpt"] = windows[best]
                document.metadata["retrieval_warning"] = "长块按模型Token窗口精排，本次引用只覆盖选中的摘录；完整内容见原页。"
            scored.append((document, window_scores[best]))
            offset += len(windows)
        if not all(math.isfinite(score) for _, score in scored):
            raise ValueError("重排模型返回非有限分数")
        # sigmoid 只将分数映射到 0~1，不代表已经校准的命中概率。
        return sorted(scored, key=lambda item: item[1], reverse=True)[:k]

    def _passages(self, query, document, tokenizer):
        """长表逐行分窗并重复表头；正文按词表偏移分窗，不让模型静默丢掉表尾。"""
        text = document.page_content
        # 短正文沿用原成对输入，普通短块不增加模型调用。
        if len(text) + len(query) < self.max_length // 4:
            return [text]
        query_tokens = len(tokenizer.encode(query, add_special_tokens=False))
        budget = self.max_length - query_tokens - tokenizer.num_special_tokens_to_add(pair=True)
        if budget <= 0:
            raise ValueError("问题超过重排模型Token窗口，请缩短问题")
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(encoded["input_ids"]) <= budget:
            return [text]
        if document.metadata.get("content_type") == "table":
            lines = text.splitlines(keepends=True)
            separator = next((i for i, line in enumerate(lines) if re.match(r"\|\s*:?-", line)), None)
            if separator is not None:
                header = "".join(lines[:separator + 1])
                windows, current = [], header
                for row in lines[separator + 1:]:
                    if len(tokenizer.encode(header + row, add_special_tokens=False)) > budget:
                        raise ValueError("表格单行和表头超过重排Token窗口，请按原页核验")
                    if len(tokenizer.encode(current + row, add_special_tokens=False)) > budget:
                        windows.append(current)
                        current = header
                    current += row
                if current != header:
                    windows.append(current)
                return windows
        offsets = encoded["offset_mapping"]
        step = max(1, budget - min(64, budget // 4))
        return [text[offsets[start][0]:offsets[min(start + budget, len(offsets)) - 1][1]]
                for start in range(0, len(offsets), step)]
