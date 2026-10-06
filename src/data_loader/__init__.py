"""模块一：文档格式分发、批量导入与状态追踪。"""

import hashlib
import logging
import tempfile
import time
from pathlib import Path
from threading import Lock, RLock

from src.data_loader.docx_loader import load_docx
from src.data_loader.pdf_loader import load_pdf
from src.data_loader.text_loader import load_text


LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_text,
           ".md": load_text, ".markdown": load_text}


# 同进程各会话共享文档锁；登记锁只保护创建，实际导入按文档分别串行。
_document_locks = {}
_document_locks_guard = Lock()


def document_lock(raw_dir: str | Path, doc_id: str):
    """同一原文的导入、删除和恢复共用锁，不影响其他文档。"""
    key = (str(Path(raw_dir).resolve()), doc_id)
    with _document_locks_guard:
        return _document_locks.setdefault(key, RLock())


def load_document(file_path: str | Path):
    """参考上游的扩展名分发，只加载文本，不执行分块或向量化。"""
    suffix = Path(file_path).suffix.lower()
    if suffix not in LOADERS:
        raise ValueError(f"不支持的文件格式：{suffix}")
    return LOADERS[suffix](file_path)


def create_import_tasks(files: list[tuple[str, bytes]]) -> list[dict]:
    """从文件名与原始字节创建任务，完全相同的重复项仅保留一份。"""
    tasks = []
    seen = set()
    for name, data in files:
        task_id = hashlib.sha256(name.encode("utf-8") + b"\0" + data).hexdigest()
        if task_id in seen:
            continue
        seen.add(task_id)
        tasks.append({"id": task_id, "name": name, "data": data,
                      "status": "pending", "attempts": 0, "error": "",
                      "elapsed": 0.0, "documents": [], "path": ""})
    return tasks


def batch_import(tasks: list[dict], raw_dir: str | Path,
                 max_file_size_mb: int = 20, retry_failed: bool = False):
    """逐文件处理，更新原任务列表并产出进度；重试时只选取失败项。

    一次调用每份目标文档最多尝试一次。成功文件存入“内容指纹/原文件名”，
    失败不阻断后续任务；临时解析文件自动清理，不留下损坏文件。
    """
    root = Path(raw_dir).expanduser().resolve()
    statuses = {"failed"} if retry_failed else {"pending", "loading"}
    targets = [task for task in tasks if task["status"] in statuses]
    total = len(targets)
    yield {"completed": 0, "total": total}
    for index, task in enumerate(targets):
        task.update(status="loading", error="", documents=[], path="")
        task["attempts"] += 1
        start = time.perf_counter()
        yield {"completed": index, "total": total}
        try:
            with document_lock(root, hashlib.sha256(task["data"]).hexdigest()):
                name, data = task["name"], task["data"]
                # 上传文件名只接受普通文件名，不能将其当作任意磁盘路径。
                if not name or name in {".", ".."} or "/" in name or "\\" in name:
                    raise ValueError("文件名不能包含目录路径")
                if Path(name).suffix.lower() not in LOADERS:
                    raise ValueError(f"不支持的文件格式：{Path(name).suffix}")
                if len(data) > max_file_size_mb * 1024 * 1024:
                    raise ValueError(f"文件过大，单份最大支持 {max_file_size_mb} MB")

                # 先在临时目录解析，通过后再保存，避免失败文件混入成功文献。
                with tempfile.TemporaryDirectory() as temporary_dir:
                    temporary_path = Path(temporary_dir) / name
                    temporary_path.write_bytes(data)
                    documents = load_document(temporary_path)

                destination = root / hashlib.sha256(data).hexdigest() / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if destination.read_bytes() != data:
                        raise ValueError("保存位置已有不同内容的文件，未覆盖")
                else:
                    saved_file = destination.open("xb")
                    try:
                        with saved_file:
                            saved_file.write(data)
                    except OSError:
                        # 只清理本次独占创建的未完成文件，下一次重试可重新保存。
                        destination.unlink(missing_ok=True)
                        raise
                for document in documents:
                    document.metadata.update(source=str(destination), source_file=name)
                task.update(status="success", documents=documents, path=str(destination))
        except Exception as error:
            # 在批量边界隔离解析/保存错误，继续处理剩余文档。
            # 服务端终端保留完整异常链，便于区分解析错误和临时目录清理错误。
            logging.getLogger(__name__).exception("文档导入失败：%s", task["name"])
            task.update(status="failed", error=f"{type(error).__name__}: {error}")
        task["elapsed"] = round(time.perf_counter() - start, 3)
        yield {"completed": index + 1, "total": total}
