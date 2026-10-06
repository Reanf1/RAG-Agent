"""为Windows的HNSW窄字符文件接口提供英文路径，原索引仍在配置目录。"""

from hashlib import sha256
import os
from pathlib import Path
import sys
import tempfile
from threading import Lock


_path_lock = Lock()


def chroma_persist_path(directory: Path) -> str:
    """返回Chroma实际访问路径；junction只增加入口，不复制、迁移或重建索引。

    中文原文和项目目录仍可使用。英文临时入口以原目录哈希命名，重启复用；
    清理临时目录后下次自动重建入口。不能再次resolve返回值，否则又回到中文路径。
    """
    directory = directory.resolve()
    if sys.platform != "win32" or str(directory).isascii():
        return str(directory)
    root = Path(tempfile.gettempdir()) / "rag-agent-index-links"
    if not str(root).isascii():
        raise ValueError("Windows索引入口需要英文临时目录；请将TEMP/TMP指向可写英文目录后重启，原索引保留")
    name = sha256(os.path.normcase(str(directory)).encode("utf-8")).hexdigest()
    alias = root / name
    with _path_lock:
        root.mkdir(parents=True, exist_ok=True)
        directory.mkdir(parents=True, exist_ok=True)
        if not alias.exists():
            # CPython的Windows原生接口支持Unicode目标，无shell拼接、无需符号链接权限。
            from _winapi import CreateJunction
            try:
                CreateJunction(str(directory), str(alias))
            except FileExistsError:
                pass  # 另一进程已建立入口，仍须核验目标相同。
        if not alias.samefile(directory):
            raise ValueError("Windows索引入口未指向同一目录，请检查临时入口冲突；不改写原索引")
    return str(alias)
