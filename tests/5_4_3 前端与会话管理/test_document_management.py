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

    def test_unindexed_list_delete_restore_without_creating_database(self):
        from src.frontend.components.documents import list_documents, delete_document, restore_document
        with patch("src.retrieval.vector_store.get_embeddings", side_effect=AssertionError("不可加载模型")):
            self.assertEqual(list_documents(self.raw, self.index)[0]["chunks"], 0)
            self.assertEqual(delete_document(self.raw, self.index, self.doc_id), 0)
            self.assertEqual(list_documents(self.raw, self.index), [])
            tasks = restore_document(self.raw, self.doc_id)
        self.assertEqual(tasks[0]["data"], self.data)
        self.assertEqual(tasks[0]["status"], "pending")
        self.assertFalse(self.index.exists())

    def test_reject_path_and_symlink(self):
        from src.frontend.components.documents import delete_document, list_documents
        with self.assertRaises(ValueError):
            delete_document(self.raw, self.index, "../outside")
        link = self.raw / ("a" * 64)
        link.symlink_to(self.folder, target_is_directory=True)
        with self.assertRaises(ValueError):
            delete_document(self.raw, self.index, link.name)
        self.assertEqual(len(list_documents(self.raw, self.index)), 1)
        self.assertTrue((self.folder / "paper.txt").exists())

    def test_archive_conflict_preserves_source(self):
        from src.frontend.components.documents import delete_document
        (self.raw / ".trash" / self.doc_id).mkdir(parents=True)
        with self.assertRaises(ValueError):
            delete_document(self.raw, self.index, self.doc_id)
        self.assertEqual((self.folder / "paper.txt").read_bytes(), self.data)

    def test_restore_rejects_changed_content_and_no_overwrite(self):
        from src.frontend.components.documents import delete_document, restore_document
        delete_document(self.raw, self.index, self.doc_id)
        archived = self.raw / ".trash" / self.doc_id / "paper.txt"
        archived.write_bytes(b"Changed")
        with self.assertRaises(ValueError):
            restore_document(self.raw, self.doc_id)
        self.assertTrue(archived.exists())
        archived.write_bytes(self.data)
        self.folder.mkdir()
        with self.assertRaises(FileExistsError):
            restore_document(self.raw, self.doc_id)
        self.assertTrue(archived.exists())

    def test_missing_source_never_deletes_index(self):
        from src.frontend.components.documents import delete_document
        with patch("src.frontend.components.documents.VectorStore") as store:
            with self.assertRaises(FileNotFoundError):
                delete_document(self.raw, self.index, "b" * 64)
            store.assert_not_called()

    def test_graph_parallel_ids_and_join(self):
        from src.frontend.components.trace import trace_graph
        graph = trace_graph([
            {"type": "thought"},
            {"type": "tool_call", "name": 'a"tool', "call_id": "a"},
            {"type": "tool_call", "name": "b", "call_id": "b"},
            {"type": "tool_result", "name": "b", "call_id": "b"},
            {"type": "tool_result", "name": "a", "call_id": "a"},
            {"type": "observation", "decision": "continue"},
            {"type": "thought"}, {"type": "done", "stop_reason": "max_iterations"}])
        for edge in ("n0 -> n1", "n0 -> n2", "n2 -> n3", "n1 -> n4", "n3 -> n5", "n4 -> n5", "n5 -> n6"):
            self.assertIn(edge, graph)
        self.assertIn('a\\"tool', graph)
        self.assertNotIn("n1 -> n2", graph)


if __name__ == "__main__":
    unittest.main()
