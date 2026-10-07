"""为Windows的HNSW窄字符文件接口提供英文路径，原索引仍在配置目录。"""

from hashlib import sha256
import os
from pathlib import Path
import sys
import tempfile
from threading import Lock
from uuid import uuid4


_path_lock = Lock()
_process_aliases = {}


def chroma_persist_path(directory: Path) -> str:
    """返回Chroma实际访问路径；junction只增加入口，不复制、迁移或重建索引。

    中文原文和项目目录仍可使用。英文临时入口以原目录哈希命名，重启复用；
    清理临时目录后下次自动重建入口。旧入口被Windows信任检查拒绝时，核验目标后
    为当前进程新建入口，保留原入口和原索引。不能resolve返回值，否则又回到中文路径。
    """
    directory = directory.resolve()
    if sys.platform != "win32" or str(directory).isascii():
        return str(directory)
    root = Path(tempfile.gettempdir()) / "rag-agent-index-links"
    if not str(root).isascii():
        raise ValueError("Windows索引入口需要英文临时目录；请将TEMP/TMP指向可写英文目录后重启，原索引保留")
    name = sha256(os.path.normcase(str(directory)).encode("utf-8")).hexdigest()
    with _path_lock:
        alias = _process_aliases.get(name, root / name)
        root.mkdir(parents=True, exist_ok=True)
        directory.mkdir(parents=True, exist_ok=True)
        # CPython的Windows原生接口支持Unicode目标，无shell拼接、无需符号链接权限。
        from _winapi import CreateJunction
        try:
            if not alias.exists():
                try:
                    CreateJunction(str(directory), str(alias))
                except FileExistsError:
                    pass  # 另一进程已建立入口，仍须核验目标相同。
            matches = alias.samefile(directory)
        except OSError as error:
            if getattr(error, "winerror", None) != 448:
                raise
            # readlink不遍历挂载点；仅允许已知目标的旧入口换用当前进程创建的入口。
            target = os.readlink(alias)
            if target.startswith("\\\\?\\"):
                target = target[4:]
            if os.path.normcase(os.path.normpath(target)) != os.path.normcase(str(directory)):
                raise ValueError("Windows索引入口未指向同一目录，请检查临时入口冲突；不改写原索引") from error
            alias = root / (name + "-" + uuid4().hex[:12])
            CreateJunction(str(directory), str(alias))
            matches = alias.samefile(directory)
            if matches:
                _process_aliases[name] = alias
        if not matches:
            raise ValueError("Windows索引入口未指向同一目录，请检查临时入口冲突；不改写原索引")
    return str(alias)
