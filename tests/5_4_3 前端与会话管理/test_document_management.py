"""5.4.3 前端与会话管理：TestDocumentManagement。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tempfile
import unittest
from unittest.mock import patch


class TestDocumentManagement(unittest.TestCase):
    """回收边界与实际加载任务验证；无需生成模型。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.raw = Path(self.directory.name) / "raw"
        self.index = Path(self.directory.name) / "index"
        self.data = b"Six encoder layers"
        import hashlib
        self.doc_id = hashlib.sha256(self.data).hexdigest()
        self.folder = self.raw / self.doc_id
        self.folder.mkdir(parents=True)
        (self.folder / "paper.txt").write_bytes(self.data)


    def test_read_complete_text_and_markdown_without_index(self):
        from src.frontend.components.documents import read_document_content
        with patch("src.frontend.components.documents.VectorStore", side_effect=AssertionError("不可读取索引")):
            parts = read_document_content(self.raw, self.doc_id)
        self.assertEqual([part.page_content for part in parts], [self.data.decode()])
        (self.folder / "paper.txt").rename(self.folder / "paper.md")
        self.assertEqual(read_document_content(self.raw, self.doc_id)[0].page_content, self.data.decode())


if __name__ == "__main__":
    unittest.main()
