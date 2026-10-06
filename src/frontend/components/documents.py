"""页面文档管理：真实块数、原文回收和重新导入，不调用生成模型。"""

from hashlib import sha256
from pathlib import Path
import re
import json

from src.data_loader import LOADERS, create_import_tasks, document_lock, load_document
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config


def list_documents(raw_dir: Path, index_dir: Path) -> list[dict]:
    """合并已保存原文和当前索引；有块不等同于全部预期块导入成功。"""
    rows, actual_ids = {}, {}
    if (index_dir / "chroma.sqlite3").is_file():
        for chunk in VectorStore(index_dir).list_chunks():
            identifier = chunk.metadata["doc_id"]
            row = rows.setdefault(identifier, {"doc_id": identifier, "name": chunk.metadata.get("source_file", "未知文档"),
                                               "chunks": 0, "source_available": False})
            row["chunks"] += 1
            actual_ids.setdefault(identifier, set()).add(chunk.metadata["chunk_id"])
    if raw_dir.exists():
        for folder in sorted(raw_dir.iterdir()):
            if folder.is_symlink() or not folder.is_dir() or not re.fullmatch(r"[0-9a-f]{64}", folder.name):
                continue
            files = [f for f in sorted(folder.iterdir()) if f.is_file() and not f.is_symlink() and f.suffix.lower() in LOADERS]
            if files:
                row = rows.setdefault(folder.name, {"doc_id": folder.name, "chunks": 0})
                row.update(name=" / ".join(f.name for f in files), source_available=True)
    for identifier, row in rows.items():
        row.update(index_status="已有索引，完整性未核验" if row["chunks"] else "未向量化", expected_chunks=None)
        manifest = raw_dir / identifier / ".index_status.json"
        if manifest.is_file():
            try:
                state = json.loads(manifest.read_text(encoding="utf-8"))
                if (state["index_directory"] == str(index_dir.resolve()) and
                        state["collection"] == load_config()["retrieval"]["collection_name"]):
                    expected = set(state["expected_ids"])
                    row["expected_chunks"] = len(expected)
                    row["index_status"] = ("已向量化" if state["complete"] and expected == actual_ids.get(identifier, set())
                                           else "部分入库" if row["chunks"] else "未向量化")
            except (OSError, ValueError, KeyError, TypeError):
                row["index_status"] = "索引状态记录异常，完整性未核验"
    return sorted(rows.values(), key=lambda row: (row["name"], row["doc_id"]))


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
        raise FileNotFoundError("原文缺失，请重新上传或恢复文档")
    # 同内容不同名称共用一个ID；验证指纹后只解析一份，避免重复展示。
    path = files[0]
    if sha256(path.read_bytes()).hexdigest() != doc_id:
        raise ValueError("原文内容已改变，与文档ID不一致，请重新上传")
    return load_document(path)


def delete_document(raw_dir: Path, index_dir: Path, doc_id: str) -> int:
    """移除检索块并回收原文；索引失败时将原文归回，允许修复后重试。"""
    with document_lock(raw_dir, doc_id):
        source = _document_directory(raw_dir, doc_id)
        if not source.is_dir():
            raise FileNotFoundError("原文缺失，无法安全回收；请先补全原文再删除")
        trash = raw_dir.resolve() / ".trash" / doc_id
        if trash.parent.is_symlink() or trash.exists():
            raise ValueError("回收位置异常或已有同ID记录，请先检查回收区")
        trash.parent.mkdir(parents=True, exist_ok=True)
        source.rename(trash)
        try:
            return VectorStore(index_dir).delete_document(doc_id) if (index_dir / "chroma.sqlite3").is_file() else 0
        except Exception:
            trash.rename(source)
            raise


def restore_document(raw_dir: Path, doc_id: str) -> list[dict]:
    """原文归回后交给现有批量导入/索引流程；恢复索引失败仍保留原文。"""
    with document_lock(raw_dir, doc_id):
        destination = _document_directory(raw_dir, doc_id)
        trash = raw_dir.resolve() / ".trash" / doc_id
        if trash.parent.is_symlink() or trash.is_symlink() or not trash.is_dir():
            raise ValueError("回收记录不存在或不是安全目录")
        if destination.exists():
            raise FileExistsError("该文档已重新上传，恢复不会覆盖现有原文")
        files = [(f.name, f.read_bytes()) for f in sorted(trash.iterdir())
                 if f.is_file() and not f.is_symlink() and f.suffix.lower() in LOADERS]
        if not files:
            raise ValueError("回收区没有可恢复的原文")
        if any(sha256(data).hexdigest() != doc_id for _, data in files):
            raise ValueError("回收原文已改变，内容指纹不一致；请检查后重新上传")
        tasks = create_import_tasks(files)
        trash.rename(destination)
        return tasks


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
        raise FileNotFoundError("引用PDF原文不可用；删除的文档需先恢复")
    data = path.read_bytes()
    if sha256(data).hexdigest() != metadata["doc_id"]:
        raise ValueError("引用原文已改变，不能用旧引用定位新文件")
    page = metadata["page_number"]
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        if type(page) is not int or not 1 <= page <= len(pdf):
            raise ValueError("引用物理页超出原文范围")
        image = pdf[page - 1].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).tobytes("png")
    return {"image": image, "pdf": data, "filename": filename, "page_number": page}
