"""读取 UTF-8 TXT/Markdown 原文，保留格式、来源和行范围。"""

import hashlib
from pathlib import Path

from langchain_core.documents import Document


def load_text(file_path: str | Path) -> list[Document]:
    """整篇返回一个 Document，后续再分块；兼容 UTF-8 BOM。

    支持 .txt、.md、.markdown，不渲染 Markdown，不猜测其他编码。
    保留缩进和原始换行；空内容明确报错，解码错误向上传递。
    """
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在或不是普通文件：{path}")
    file_type = path.suffix.lower()
    if file_type not in {".txt", ".md", ".markdown"}:
        raise ValueError(f"纯文本加载器不支持此文件格式：{path.suffix}")

    data = path.read_bytes()
    # 从原始字节解码，避免文本文件读取时自动改写 CRLF 或 Markdown 缩进。
    text = data.decode("utf-8-sig")
    if not text.strip():
        raise ValueError(f"纯文本文件没有有效内容：{path.name}")

    return [Document(page_content=text, metadata={
        "source": str(path),
        "source_file": path.name,
        "file_type": file_type,
        "doc_id": hashlib.sha256(data).hexdigest(),
        "line_start": 1,
        "line_end": len(text.splitlines()),
    })]
