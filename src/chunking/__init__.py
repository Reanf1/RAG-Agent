"""三种分块的统一入口，以及共用的来源元数据处理。"""

import hashlib
import json
import re
from bisect import bisect_right
from copy import deepcopy

from langchain_core.documents import Document

from src.utils.config import load_config


def split_documents(documents: list[Document], strategy: str | None = None,
                    chunk_size: int | None = None,
                    chunk_overlap: int | None = None) -> list[Document]:
    """未指定的参数从 config.yaml 读取，返回可供后续检索使用的文档块。"""
    from .fixed_chunk import split_fixed
    from .recursive_chunk import split_recursive
    from .semantic_chunk import split_semantic

    if strategy is None or chunk_size is None or chunk_overlap is None:
        config = load_config()["chunking"]
        strategy = config["strategy"] if strategy is None else strategy
        chunk_size = config["chunk_size"] if chunk_size is None else chunk_size
        chunk_overlap = config["chunk_overlap"] if chunk_overlap is None else chunk_overlap
    strategies = {"fixed": split_fixed, "recursive": split_recursive, "semantic": split_semantic}
    if strategy not in strategies:
        raise ValueError(f"不支持的分块策略：{strategy}，请选择 fixed、recursive 或 semantic")
    return strategies[strategy](documents, chunk_size, chunk_overlap)


def _validate_parameters(chunk_size: int, chunk_overlap: int):
    """重叠必须小于块大小，保证每次切分都能向前推进。"""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数（字符数）")
    if type(chunk_overlap) is not int or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap 必须为整数，且满足 0 <= chunk_overlap < chunk_size")


def _build_chunks(documents, split_ranges, strategy, chunk_size, chunk_overlap):
    """用原文区间构建块；三种策略共用表格保护、位置和稳定 ID。"""
    _validate_parameters(chunk_size, chunk_overlap)
    chunks = []
    for document in documents:
        text = document.page_content
        if not text.strip():
            continue
        metadata = document.metadata
        is_table = metadata.get("content_type") == "table" or metadata.get("block_type") == "table"
        # 独立表格不从行或单元格中间切开，保留跨页行来源；允许超过目标大小。
        ranges = [(0, len(text))] if is_table else split_ranges(text)
        parent_key = [metadata.get("doc_id", metadata.get("source", "")),
                      metadata.get("page"), metadata.get("block_index"), metadata.get("table_id"), text]
        parent_id = hashlib.sha256(json.dumps(parent_key, ensure_ascii=False).encode("utf-8")).hexdigest()
        # 与 str.splitlines() 一致；CRLF 作为一个换行，换行字符归前一行。
        line_ends = [match.end() for match in re.finditer(r"\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]", text)]
        layout, layout_origin = _layout_ranges(metadata.get("formula_layout"), text), None
        for index, (start, end) in enumerate(ranges):
            content = text[start:end]
            if not content.strip():
                continue
            chunk_metadata = deepcopy(metadata)
            chunk_metadata.update({
                "chunk_id": hashlib.sha256(
                    f"{parent_id}:{strategy}:{chunk_size}:{chunk_overlap}:{start}:{end}".encode("utf-8")
                ).hexdigest(),
                "chunk_index": index, "chunk_strategy": strategy,
                "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                "start_index": start, "end_index": end,
            })
            if layout is not None:
                # 跨块行保留完整坐标；重复文本保守匹配全部位置，不推测排版对应关系。
                entries = [entry for entry, positions in layout if
                           any(left < end and right > start for left, right in positions)
                           or (not positions and layout_origin is None)]
                chunk_metadata["formula_layout"] = json.dumps(entries, ensure_ascii=False)
                if layout_origin is not None and any(not positions for _, positions in layout):
                    chunk_metadata["formula_layout_origin_chunk_id"] = layout_origin
                # 不在正文中的表格等坐标留在首块，其他块提供来源指针；原PDF仍完整。
                layout_origin = layout_origin or chunk_metadata["chunk_id"]
            if "line_start" in metadata:
                chunk_metadata["line_start"] = metadata["line_start"] + bisect_right(line_ends, start)
                chunk_metadata["line_end"] = metadata["line_start"] + bisect_right(line_ends, end - 1)
            if is_table:
                chunk_metadata["chunk_preserved"] = "table"
            chunks.append(Document(page_content=content, metadata=chunk_metadata))
    return chunks


def _layout_ranges(raw, text):
    """一次定位本页布局行；旧格式无法识别时保持原元数据，不能静默丢弃。"""
    if raw is None:
        return None
    try:
        lines = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(lines, list) or any(not isinstance(line, dict) or not isinstance(line.get("text"), str) for line in lines):
        return None
    layout = []
    for line in lines:
        positions = [(match.start(), match.end()) for match in re.finditer(re.escape(line["text"]), text)] if line["text"] else []
        layout.append((line, positions))
    return layout
