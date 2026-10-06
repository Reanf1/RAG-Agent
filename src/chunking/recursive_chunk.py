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
            # 起点和终点均须前进；短分隔符不能再次匹配到已覆盖的旧片段。
            lower = max(previous_start + 1, previous_end - chunk_overlap,
                        previous_end - len(content) + 1)
            start = text.find(content, lower)
            if start < 0:
                raise ValueError("递归分块无法定位原文，已停止生成错误来源")
            end = start + len(content)
            ranges.append((start, end))
            previous_start, previous_end = start, end
        return ranges

    return _build_chunks(documents, split_ranges, "recursive", chunk_size, chunk_overlap)
