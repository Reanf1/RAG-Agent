"""页面文档管理：原文读取与确认后的删除，不调用生成模型。"""

from hashlib import sha256
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import re
import shutil

from src.data_loader import LOADERS, document_lock, load_document
from src.retrieval.vector_store import VectorStore, list_documents


def _document_directory(raw_dir: Path, doc_id: str) -> Path:
    """只接受上传内容指纹，不能将用户输入当作任意磁盘路径。"""
    if not re.fullmatch(r"[0-9a-f]{64}", doc_id):
        raise ValueError("文档ID必须为64位内容指纹")
    directory = raw_dir.resolve() / doc_id
    if directory.is_symlink():
        raise ValueError("文档目录不能是符号链接")
    return directory


def read_document_content(raw_dir: Path, doc_id: str):
    """读取已保存原文的完整解析内容，不读取检索块、不重新向量化。"""
    directory = _document_directory(raw_dir, doc_id)
    files = [path for path in sorted(directory.iterdir())
             if path.is_file() and not path.is_symlink() and path.suffix.lower() in LOADERS] if directory.is_dir() else []
    if not files:
        raise FileNotFoundError("原文缺失，请重新上传")
    # 同内容不同名称共用一个ID；验证指纹后只解析一份，避免重复展示。
    path = files[0]
    if sha256(path.read_bytes()).hexdigest() != doc_id:
        raise ValueError("原文内容已改变，与文档ID不一致，请重新上传")
    return deepcopy(_parse_original(str(path.resolve()), doc_id))


@lru_cache(maxsize=8)
def _parse_original(path: str, doc_id: str):
    """缓存解析计算，不缓存文件存在性或指纹检查；返回前复制避免页面修改缓存。"""
    parts = load_document(path)
    if any(part.metadata["doc_id"] != doc_id for part in parts):
        raise ValueError("解析期间原文内容已改变，请重新上传")
    return parts


def delete_document(raw_dir: Path, index_dir: Path, doc_id: str) -> int:
    """先删除索引，再删除应用保存的原文副本；调用方负责确认，不提供恢复。"""
    with document_lock(raw_dir, doc_id):
        source = _document_directory(raw_dir, doc_id)
        removed = VectorStore(index_dir).delete_document(doc_id) if (index_dir / "chroma.sqlite3").is_file() else 0
        if source.is_dir():
            shutil.rmtree(source)
        return removed


def read_pdf_page(raw_dir: Path, reference: dict) -> dict:
    """按已上传指纹与物理页读取原文，不采用引用中的任意文件路径。"""
    import pymupdf
    metadata = reference["metadata"]
    directory = _document_directory(raw_dir, metadata["doc_id"])
    filename = reference["source_file"]
    if Path(filename).name != filename:
        raise ValueError("引用文件名不能包含路径")
    path = directory / filename
    if path.is_symlink() or path.suffix.lower() != ".pdf" or not path.is_file():
        raise FileNotFoundError("引用PDF原文不可用；删除的文档需重新上传")
    data = path.read_bytes()
    if sha256(data).hexdigest() != metadata["doc_id"]:
        raise ValueError("引用原文已改变，不能用旧引用定位新文件")
    page = metadata["page_number"]
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        if type(page) is not int or not 1 <= page <= len(pdf):
            raise ValueError("引用物理页超出原文范围")
        image = pdf[page - 1].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).tobytes("png")
    return {"image": image, "pdf": data, "filename": filename, "page_number": page}
