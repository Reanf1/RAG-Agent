"""精简后的基本约定：删除、共享文献状态和单批摘要。只使用临时数据。"""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.agent.memory import MemoryManager
from src.agent.tools import paper_list
from src.data_loader import create_import_tasks
from src.frontend.components.documents import delete_document
from src.retrieval.vector_store import VectorStore, batch_build_index, list_documents
from src.utils.config import load_config
from tests.helpers import SmallEmbeddings


class TestBasicSimplification(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.raw, self.index = self.root / "raw", self.root / "index"
        self.config = load_config()
        self.config["paths"].update(raw_documents=str(self.raw), vector_index=str(self.index))
        self.config["memory"].update(summary_trigger_turns=4, summary_keep_recent_turns=2)
        for module in ("src.agent.tools", "src.agent.memory", "src.retrieval.vector_store"):
            setting = patch(module + ".load_config", return_value=self.config)
            setting.start()
            self.addCleanup(setting.stop)
        self.memory = MemoryManager(self.root / "sessions.sqlite3")

    def ingest(self):
        store = VectorStore(self.index, SmallEmbeddings())
        tasks = create_import_tasks([("论文.txt", "方法：农业图像分类。结果：准确率91%。".encode())])
        list(batch_build_index(tasks, self.raw, vector_store=store))
        self.assertTrue(tasks[0]["indexed"])
        return store, tasks[0]["documents"][0].metadata["doc_id"]

    def test_document_delete_removes_copy_and_index(self):
        store, identifier = self.ingest()
        self.assertEqual(delete_document(self.raw, self.index, identifier), 1)
        self.assertEqual(store.count(), 0)
        self.assertFalse((self.raw / identifier).exists())
        self.assertEqual(list_documents(self.raw, self.index), [])

    def test_document_list_and_tool_share_status(self):
        _, identifier = self.ingest()
        row = list_documents(self.raw, self.index)[0]
        tool_row = paper_list.invoke({})["papers"][0]
        self.assertEqual((row["doc_id"], row["chunks"], row["index_status"]),
                         (tool_row["doc_id"], tool_row["indexed_chunks"], tool_row["index_status"]))
        state = json.loads((self.raw / identifier / ".index_status.json").read_text())
        state.update(expected_chunks=2, complete=False)
        (self.raw / identifier / ".index_status.json").write_text(json.dumps(state))
        self.assertEqual(list_documents(self.raw, self.index)[0]["index_status"], "部分入库")

    def test_session_delete_removes_messages_and_summary(self):
        session = self.memory.create_session("甲")
        other = self.memory.create_session("乙")
        self.memory.append_turn("甲", session, "问题", "回答")
        with sqlite3.connect(self.memory.db_path) as connection:
            connection.execute("INSERT INTO summaries VALUES(?, ?, ?)", (session, "旧摘要", 2))
        self.memory.delete_session("甲", session)
        self.assertEqual(self.memory.list_sessions("甲"), [])
        self.assertEqual(self.memory.list_sessions("乙"), [other])
        with self.assertRaises(LookupError):
            self.memory.get_messages("甲", session)
        with sqlite3.connect(self.memory.db_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM summaries").fetchone()[0], 0)

    def test_summary_uses_one_batch_and_keeps_recent_pairs(self):
        session = self.memory.create_session("甲")
        for index in range(6):
            self.memory.append_turn("甲", session, f"问题{index}", f"回答{index}")
        result = {"summary": "研究农业图像分类。", "usage": {"prompt_eval_count": 100, "eval_count": 10},
                  "elapsed_seconds": 0.1}
        with patch("src.agent.memory._summarize", return_value=result) as summarize:
            context = self.memory.get_context("甲", session)
        summarize.assert_called_once()
        self.assertEqual(context["summary"], result["summary"])
        self.assertEqual(len(context["history"]), 4)
        self.assertEqual(len(self.memory.get_messages("甲", session)), 12)


if __name__ == "__main__":
    unittest.main()
