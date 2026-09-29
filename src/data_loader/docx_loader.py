"""使用 python-docx 按正文顺序读取 Word 段落和表格。"""

import hashlib
from pathlib import Path

from docx import Document as WordDocument
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document


def load_docx(file_path: str | Path) -> list[Document]:
    """每个非空段落或顶层表格返回一个 Document，不生成物理页码。

    表格按行拼接，以制表符分隔单元格；保留正文块、段落或表格的
    原始序号，空块跳过后不重新编号。暂不读取嵌套表格、页眉页脚和图片。
    """
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在或不是普通文件：{path}")
    if path.suffix.lower() != ".docx":
        raise ValueError(f"Word 加载器仅支持 .docx，请先转换文件格式：{path.suffix}")

    word = WordDocument(str(path))
    metadata = {
        "source": str(path),
        "source_file": path.name,
        "file_type": ".docx",
        "doc_id": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    documents = []
    paragraph_index = table_index = 0
    # 顺序遍历正文，避免分别读取 paragraphs/tables 导致表格位置丢失。
    for block_index, block in enumerate(word.iter_inner_content(), 1):
        if isinstance(block, Paragraph):
            paragraph_index += 1
            text = block.text.strip()
            location = {"block_type": "paragraph", "paragraph_index": paragraph_index}
        else:
            table_index += 1
            text = "\n".join(
                "\t".join(cell.text.strip() for cell in row.cells)
                for row in block.rows
            )
            location = {"block_type": "table", "table_index": table_index}

        if not text.strip():
            continue
        documents.append(Document(page_content=text, metadata={
            **metadata,
            "block_index": block_index,
            **location,
        }))

    if not documents:
        raise ValueError(f"Word 正文和表格未提取到文本：{path.name}")
    return documents
