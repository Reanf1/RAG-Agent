"""Windows中文目录通过英文junction访问同一份索引，禁止复制或换空库。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.utils.chroma_path import chroma_persist_path


class TestChromaPath(unittest.TestCase):

    def test_windows_ascii_path_needs_no_alias(self):
        with tempfile.TemporaryDirectory() as root, patch("src.utils.chroma_path.sys.platform", "win32"):
            self.assertEqual(chroma_persist_path(Path(root)), str(Path(root).resolve()))

    def test_unicode_alias_is_same_index_and_reused(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "工作" / "index"
            directory.mkdir(parents=True)
            original = directory / "header.bin"
            original.write_bytes(b"original-index")

            # Mac用符号链接模拟junction，只验证路径与索引同一性，不冒充Windows实测。
            def junction(source, destination):
                Path(destination).symlink_to(source, target_is_directory=True)

            native = SimpleNamespace(CreateJunction=junction)
            with patch("src.utils.chroma_path.sys.platform", "win32"), \
                    patch("src.utils.chroma_path.tempfile.gettempdir", return_value=root), \
                    patch.dict("sys.modules", {"_winapi": native}):
                alias = chroma_persist_path(directory)
                self.assertTrue(alias.isascii())
                self.assertEqual(Path(alias).resolve(), directory.resolve())
                self.assertEqual(chroma_persist_path(directory), alias)
                (Path(alias) / "new.bin").write_bytes(b"new-vector")
            self.assertEqual(original.read_bytes(), b"original-index")
            self.assertEqual((directory / "new.bin").read_bytes(), b"new-vector")
