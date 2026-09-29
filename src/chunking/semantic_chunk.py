"""按段落和句子边界切分，不加载 Embedding 或其他语义模型。"""

import re

from langchain_core.documents import Document

from . import _build_chunks
from .fixed_chunk import _fixed_ranges


def _semantic_ranges(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """短段落作为整体，长段落拆句，超长单句按字符兜底。"""
    units = []
    start = 0
    paragraph_ends = [match.end() for match in re.finditer(r"(?:\r?\n[ \t]*){2,}", text)]
    for end in paragraph_ends + [len(text)]:
        if end == start:
            continue
        if end - start <= chunk_size:
            units.append((start, end))
        else:
            sentence_start = start
            # 英文句点后需为空白或段尾，避免将 3.14、DOI 中的点当作句号。
            boundary = r'''[。！？!?]+[”’"')）\]]*|\.[”’"')）\]]*(?=\s|$)'''
            sentence_ends = [start + match.end() for match in re.finditer(boundary, text[start:end])]
            for sentence_end in sentence_ends + [end]:
                if sentence_end == sentence_start:
                    continue
                if sentence_end - sentence_start <= chunk_size:
                    units.append((sentence_start, sentence_end))
                else:
                    # 兜底片段先不重叠，随后统一在完整单元边界选择重叠。
                    units.extend((sentence_start + left, sentence_start + right)
                                 for left, right in _fixed_ranges(text[sentence_start:sentence_end], chunk_size, 0))
                sentence_start = sentence_end
        start = end

    ranges = []
    first = 0
    while first < len(units):
        last = first
        while last + 1 < len(units) and units[last + 1][1] - units[first][0] <= chunk_size:
            last += 1
        ranges.append((units[first][0], units[last][1]))
        if last == len(units) - 1:
            break
        # 只复用完整尾部单元；同时为下一单元留空间，且每轮必须向前推进。
        following = last + 1
        while following > first + 1:
            candidate = following - 1
            if (units[last][1] - units[candidate][0] > chunk_overlap
                    or units[last + 1][1] - units[candidate][0] > chunk_size):
                break
            following = candidate
        first = following
    return ranges


def split_semantic(documents: list[Document], chunk_size: int,
                   chunk_overlap: int = 0) -> list[Document]:
    """句段切分，重叠只取完整单元，可能小于目标或为零；表格不拆。"""
    return _build_chunks(documents, lambda text: _semantic_ranges(text, chunk_size, chunk_overlap),
                         "semantic", chunk_size, chunk_overlap)
