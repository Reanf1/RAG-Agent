"""已实现按相关性拼接与字符预算截断；模型生成、引用核验及降级待接入。"""

import math
import re

from langchain_core.documents import Document

from src.generation.prompt_template import NO_CONTEXT_TEXT, build_rag_messages
from src.utils.config import load_config


def build_context(question: str, results: list[tuple[Document, float]]) -> dict:
    """把同一检索方式的 Document/分数列表组织为参考项目风格的 Context。

    分数越高越相关，同分保持输入顺序；只读原 Document。不调用检索或模型。
    字符预算包含来源和分隔符，并随系统规范及问题长度动态缩小；不是 Token 计数。
    """
    config = load_config()["generation"]
    for key in ("max_context_chars", "max_prompt_chars"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    # 使用真实模板计算固定开销，避免 Prompt 更新后长度预算失效。
    empty_messages = build_rag_messages(question)
    empty_prompt_chars = sum(len(message.content) for message in empty_messages)
    if empty_prompt_chars > config["max_prompt_chars"]:
        raise ValueError("用户问题与系统规范超过 Prompt 字符预算，请缩短问题或调整预算")
    fixed_chars = empty_prompt_chars - len(NO_CONTEXT_TEXT)
    budget = min(config["max_context_chars"], config["max_prompt_chars"] - fixed_chars)

    # 空白正文不作为依据；负分仍可排序，不设置未经评测的相关性阈值。
    candidates = [(document, score) for document, score in results if document.page_content.strip()]
    if not all(math.isfinite(score) for _, score in candidates):
        raise ValueError("检索分数必须为有限数值")
    candidates.sort(key=lambda item: item[1], reverse=True)
    parts, sources = [], []
    context_chars = 0
    partial_count = 0
    separator = "\n\n---\n\n"
    truncation_marker = "\n[正文已截断]"

    for document, _ in candidates:
        metadata = document.metadata
        filename = metadata.get("source_file") or "来源信息未提供"
        # 只使用加载器的真实位置；Word/文本没有物理页码，不推造页码。
        if "page_number" in metadata:
            location = f"第{metadata['page_number']}页（物理页码）"
            if metadata.get("page_end", metadata["page_number"]) != metadata["page_number"]:
                location = f"第{metadata['page_number']}–{metadata['page_end']}页（物理页码）"
        elif "paragraph_index" in metadata:
            location = f"段落{metadata['paragraph_index']}"
        elif "table_index" in metadata:
            location = f"表格{metadata['table_index']}"
        elif "line_start" in metadata:
            location = f"行{metadata['line_start']}–{metadata['line_end']}"
        else:
            location = "位置未记录"
        header = f"[参考文档{len(parts) + 1} - 来源: {filename}；原始块位置: {location}]\n"
        join_chars = len(separator) if parts else 0
        body_budget = budget - context_chars - join_chars - len(header)
        text = document.page_content
        partial = len(text) > body_budget
        if partial:
            # 不能只放来源而没有正文，也不能截断来源标记。尝试后续较短来源的块。
            limit = body_budget - len(truncation_marker)
            if limit <= 0:
                continue
            prefix = text[:limit].rstrip()
            boundaries = list(re.finditer(r"\n|[。！？]|[.!?](?=\s)", prefix))
            # 只在后半段寻找自然边界，避免为很早的句号浪费大部分空间。
            if boundaries and boundaries[-1].end() >= limit / 2:
                prefix = prefix[:boundaries[-1].end()].rstrip()
            if not prefix.strip():
                continue
            text = prefix + truncation_marker
        parts.append(header + text)
        context_chars += join_chars + len(parts[-1])
        partial_count += partial
        if metadata.get("source_file") and filename not in sources:
            sources.append(filename)

    if candidates and not parts:
        # 有检索结果却放不下，不把预算错误伪装成“知识库没有相关文档”。
        raise ValueError("上下文字符预算不足以容纳来源标记与正文，请调整预算或缩短问题")
    context = separator.join(parts)
    dropped_count = len(candidates) - len(parts)
    return {
        "context": context,
        "sources": sources,
        "chunk_count": len(parts),
        "context_budget_chars": budget,
        "context_chars": len(context),
        "prompt_chars": fixed_chars + len(context or NO_CONTEXT_TEXT),
        "truncated": bool(partial_count or dropped_count),
        "dropped_count": dropped_count,
    }
