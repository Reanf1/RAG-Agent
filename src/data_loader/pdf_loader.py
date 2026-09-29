"""使用 PyMuPDF 按页加载 PDF，保留后续分块和引用所需的来源信息。"""

import hashlib
import logging
from pathlib import Path

import pymupdf
from langchain_core.documents import Document


def load_pdf(file_path: str | Path) -> list[Document]:
    """返回有文本页面的 Document 列表，不执行 OCR 或结构化表格识别。

    page 为从 0 开始的索引，page_number 为从 1 开始的物理页码。
    无文本页记录警告并跳过，全部无文本或需要密码时明确报错；
    损坏文件的解析异常直接向上传递，由后续批量导入层记录失败。
    """
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在或不是普通文件：{path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF 加载器不支持此文件格式：{path.suffix}")

    documents = []
    # 上下文管理器确保加载成功或发生异常时都会关闭 PDF 文件。
    with pymupdf.open(path) as pdf:
        if not pdf.is_pdf:
            raise ValueError(f"文件内容不是 PDF：{path.name}")
        if pdf.needs_pass:
            raise ValueError(f"PDF 需要密码，请先解密后导入：{path.name}")

        # 同一文件的所有页共用内容指纹，重复加载时保持文档标识稳定。
        doc_id = hashlib.sha256(path.read_bytes()).hexdigest()
        for page in pdf:
            text = page.get_text("text", sort=True).strip()
            if not text:
                logging.getLogger(__name__).warning(
                    "%s 第 %d 页未提取到文本，已跳过；可能为空白或图片页，本加载器不执行 OCR。",
                    path.name, page.number + 1,
                )
                continue

            documents.append(Document(page_content=text, metadata={
                "source": str(path),
                "source_file": path.name,
                "file_type": ".pdf",
                "doc_id": doc_id,
                "page": page.number,
                "page_number": page.number + 1,
                "total_pages": len(pdf),
            }))

    if not documents:
        raise ValueError(f"PDF 未提取到文本，请检查是否为空白或扫描件；本加载器不执行 OCR：{path.name}")
    return documents
