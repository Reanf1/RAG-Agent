"""Windows中文目录通过英文junction访问同一份索引，禁止复制或换空库。"""

import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.utils.chroma_path import chroma_persist_path


class TestChromaPath(unittest.TestCase):
    def test_other_platform_keeps_original_path(self):
        with tempfile.TemporaryDirectory() as root, patch("src.utils.chroma_path.sys.platform", "darwin"):
            directory = Path(root) / "中文索引"
            self.assertEqual(chroma_persist_path(directory), str(directory.resolve()))
            self.assertFalse(directory.exists())

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

    def test_unicode_temp_path_is_rejected_without_creating_index(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "中文" / "index"
            with patch("src.utils.chroma_path.sys.platform", "win32"), \
                    patch("src.utils.chroma_path.tempfile.gettempdir", return_value=str(Path(root) / "用户")):
                with self.assertRaisesRegex(ValueError, "英文"):
                    chroma_persist_path(directory)
            self.assertFalse(directory.exists())

    def test_alias_conflict_never_opens_another_index(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "中文索引"
            native = SimpleNamespace(CreateJunction=lambda src, dst: Path(dst).mkdir())
            with patch("src.utils.chroma_path.sys.platform", "win32"), \
                    patch("src.utils.chroma_path.tempfile.gettempdir", return_value=root), \
                    patch.dict("sys.modules", {"_winapi": native}):
                with self.assertRaisesRegex(ValueError, "同一"):
                    chroma_persist_path(directory)

    def test_untrusted_alias_uses_new_verified_entry_without_removing_old(self):
        """Windows 448只更换当前进程入口；原入口及原索引都保留。"""
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "中文索引"
            directory.mkdir()
            (directory / "original.bin").write_bytes(b"index")
            created = []
            def junction(source, destination):
                Path(destination).symlink_to(source, target_is_directory=True)
                created.append(Path(destination))
            exists = Path.exists
            def untrusted(path):
                if created and path == created[0]:
                    error = OSError("不受信任的装入点")
                    error.winerror = 448
                    raise error
                return exists(path)
            native = SimpleNamespace(CreateJunction=junction)
            with patch("src.utils.chroma_path.sys.platform", "win32"), \
                    patch("src.utils.chroma_path.tempfile.gettempdir", return_value=root), \
                    patch.dict("sys.modules", {"_winapi": native}):
                old = Path(chroma_persist_path(directory))
                with patch.object(Path, "exists", untrusted):
                    replacement = Path(chroma_persist_path(directory))
                    self.assertEqual(chroma_persist_path(directory), str(replacement))
                self.assertNotEqual(old, replacement)
                self.assertEqual(len(created), 2)
                self.assertTrue(replacement.samefile(directory))
                self.assertTrue(Path(os.readlink(old)).samefile(directory))
                self.assertEqual((replacement / "original.bin").read_bytes(), b"index")

    def test_untrusted_conflicting_target_is_rejected(self):
        """入口报448仍须核验不跟随的目标字符串，不能换入口后掩盖冲突。"""
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "中文索引"
            error = OSError("不受信任的装入点")
            error.winerror = 448
            with patch("src.utils.chroma_path.sys.platform", "win32"), \
                    patch("src.utils.chroma_path.tempfile.gettempdir", return_value=root), \
                    patch.object(Path, "exists", side_effect=error), \
                    patch("src.utils.chroma_path.os.readlink", return_value=str(Path(root) / "other")), \
                    patch.dict("sys.modules", {"_winapi": SimpleNamespace(CreateJunction=lambda *args: None)}):
                with self.assertRaisesRegex(ValueError, "同一"):
                    chroma_persist_path(directory)

    def test_other_path_error_is_not_treated_as_untrusted_mount(self):
        with tempfile.TemporaryDirectory() as root:
            error = OSError("权限不足")
            error.winerror = 5
            with patch("src.utils.chroma_path.sys.platform", "win32"), \
                    patch("src.utils.chroma_path.tempfile.gettempdir", return_value=root), \
                    patch.object(Path, "exists", side_effect=error), \
                    patch.dict("sys.modules", {"_winapi": SimpleNamespace(CreateJunction=lambda *args: None)}):
                with self.assertRaises(OSError) as caught:
                    chroma_persist_path(Path(root) / "中文索引")
                self.assertIs(caught.exception, error)
