"""沿用参考项目的 RecursiveCharacterTextSplitter 和分隔符顺序。"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import _build_chunks, _validate_parameters


def split_recursive(documents: list[Document], chunk_size: int,
                    chunk_overlap: int = 0) -> list[Document]:
    """优先段落、换行和标点，最后按字符；重叠是目标上限。"""
    _validate_parameters(chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        length_function=len, keep_separator=True,
        strip_whitespace=False,
    )

    def split_ranges(text):
        # 保留分隔符和空白，使返回文本能直接定位到父 Document 的原文区间。
        ranges = []
        previous_start, previous_end = -1, 0
        for content in splitter.split_text(text):
            # 上游的位置查找在重复段落、大重叠时可能停留原位；至少推进一个字符。
            start = text.find(content, max(previous_start + 1, previous_end - chunk_overlap))
            end = start + len(content)
            ranges.append((start, end))
            previous_start, previous_end = start, end
        return ranges

    return _build_chunks(documents, split_ranges, "recursive", chunk_size, chunk_overlap)
