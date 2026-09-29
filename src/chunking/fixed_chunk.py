"""按字符窗口切分，可用 256、512、1024 等大小进行对比。"""

from langchain_core.documents import Document

from . import _build_chunks


def _fixed_ranges(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """窗口步长为大小减重叠，最后一块覆盖末尾后立即停止。"""
    ranges = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        ranges.append((start, end))
        if end == len(text):
            break
        start += chunk_size - chunk_overlap
    return ranges


def split_fixed(documents: list[Document], chunk_size: int,
                chunk_overlap: int = 0) -> list[Document]:
    """固定字符切分；正文相邻块精确重叠，独立表格整块保留。"""
    return _build_chunks(documents, lambda text: _fixed_ranges(text, chunk_size, chunk_overlap),
                         "fixed", chunk_size, chunk_overlap)
