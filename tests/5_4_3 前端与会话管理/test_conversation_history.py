"""5.4.3 前端与会话管理：TestConversationHistory。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from src.agent.memory import MemoryManager, run_session
from tests.helpers import isolated_agent_logs


def setUpModule():
    """保持既有 Agent 测试的模块级日志隔离。"""
    global _log_context
    _log_context = isolated_agent_logs()
    _log_context.__enter__()


def tearDownModule():
    _log_context.__exit__(None, None, None)


class TestConversationHistory(unittest.TestCase):
    """会话归属、回收及旧SQLite迁移，数据库和文件均实际执行。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "history.sqlite3"
        self.memory = MemoryManager(self.path)
        self.a = self.memory.create_session("alice")
        self.b = self.memory.create_session("bob")

    def test_archive_restore_keeps_turn_citations_and_summary(self):
        details = {"task_complete": False, "stop_reason": "incomplete", "event": {"type": "done"}}
        self.memory.append_turn("alice", self.a, "问题", "部分答案", details=details)
        rag = {"question": "层数", "answer": "6层", "citations": [{"source_file": "论文.pdf", "location": "第3页"}]}
        self.memory.append_rag_message("alice", self.a, rag)
        with sqlite3.connect(self.path) as connection:
            last = connection.execute("SELECT MAX(id) FROM messages WHERE session_id=?", (self.a,)).fetchone()[0]
            connection.execute("INSERT INTO summaries VALUES(?, ?, ?)", (self.a, "已压缩历史", last))
        self.memory.delete_session("alice", self.a)
        self.assertEqual(self.memory.list_sessions("alice"), [])
        self.assertEqual(self.memory.list_sessions("alice", archived=True), [self.a])
        self.assertEqual(self.memory.list_sessions("bob"), [self.b])
        self.memory = MemoryManager(self.path)
        self.memory.restore_session("alice", self.a)
        self.assertEqual(self.memory.get_messages("alice", self.a)[1].additional_kwargs, details)
        self.assertEqual(self.memory.get_rag_messages("alice", self.a), [rag])
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT content FROM summaries WHERE session_id=?", (self.a,)).fetchone()[0], "已压缩历史")

    def test_deleted_session_rejects_reads_writes_and_generation(self):
        self.memory.delete_session("alice", self.a)
        for operation in (lambda: self.memory.get_messages("alice", self.a),
                          lambda: self.memory.get_rag_messages("alice", self.a),
                          lambda: self.memory.append_turn("alice", self.a, "问", "答"),
                          lambda: self.memory.append_rag_message("alice", self.a, {}),
                          lambda: self.memory.clear_rag_messages("alice", self.a),
                          lambda: list(run_session("问", "alice", self.a, memory=self.memory))):
            with self.subTest(operation=operation), patch("src.agent.react_loop.run_react") as model:
                with self.assertRaises(LookupError):
                    operation()
                model.assert_not_called()
        for operation in (self.memory.delete_session, self.memory.restore_session):
            with self.assertRaises(PermissionError):
                operation("bob", self.a)
        self.assertEqual(self.memory.list_sessions("alice", archived=True), [self.a])

    def test_archive_title_keeps_owner_check_and_supports_agent_rag_empty(self):
        """归档仅开放本用户标题；完整历史和执行仍要求先恢复。"""
        self.assertEqual(self.memory.get_session_title("alice", self.a), "新会话")
        self.memory.append_rag_message("alice", self.a, {"question": "RAG论文问题"})
        self.assertEqual(self.memory.get_session_title("alice", self.a), "RAG论文问题")
        self.memory.append_turn("alice", self.a, "Agent论文问题", "答案")
        self.memory.delete_session("alice", self.a)
        self.assertEqual(self.memory.get_session_title("alice", self.a), "Agent论文问题")
        with self.assertRaises(PermissionError):
            self.memory.get_session_title("bob", self.a)
        with self.assertRaises(LookupError):
            self.memory.get_messages("alice", self.a)

    def test_old_database_migration_keeps_ids_messages_and_summary(self):
        old = Path(self.directory.name) / "old.sqlite3"
        with sqlite3.connect(old) as connection:
            connection.executescript("""
                CREATE TABLE sessions(session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL);
                CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT);
                CREATE TABLE summaries(session_id TEXT PRIMARY KEY, content TEXT, through_message_id INTEGER);
                INSERT INTO sessions VALUES('old-session', 'alice');
                INSERT INTO messages(session_id,role,content) VALUES('old-session','human','旧问题'),('old-session','ai','旧答案');
                INSERT INTO summaries VALUES('old-session','旧摘要',2);
            """)
        for _ in range(2):
            memory = MemoryManager(old)
            self.assertEqual(memory.list_sessions("alice"), ["old-session"])
            self.assertEqual([m.content for m in memory.get_messages("alice", "old-session")], ["旧问题", "旧答案"])
            self.assertEqual(memory.get_messages("alice", "old-session")[1].additional_kwargs, {})
        with sqlite3.connect(old) as connection:
            self.assertEqual(connection.execute("SELECT content,through_message_id FROM summaries").fetchone(), ("旧摘要", 2))

    def test_rag_and_trace_do_not_enter_agent_context_and_clear_is_scoped(self):
        self.memory.append_turn("alice", self.a, "Agent问题", "Agent答案", details={"event": {"tool_data": "旧工具秘密"}})
        self.memory.append_rag_message("alice", self.a, {"question": "RAG问题", "answer": "RAG答案", "citations": ["原文"]})
        self.memory.append_rag_message("bob", self.b, {"question": "另一用户问题"})
        context = self.memory.get_context("alice", self.a)
        self.assertEqual(context["history"], [{"role": "human", "content": "Agent问题"}, {"role": "ai", "content": "Agent答案"}])
        self.assertNotIn("旧工具秘密", json.dumps(context, ensure_ascii=False))
        returned = self.memory.get_rag_messages("alice", self.a)
        returned[0]["citations"].clear()
        self.assertEqual(self.memory.get_rag_messages("alice", self.a)[0]["citations"], ["原文"])
        with self.assertRaises(PermissionError):
            self.memory.clear_rag_messages("bob", self.a)
        self.memory.clear_rag_messages("alice", self.a)
        self.assertEqual(self.memory.get_rag_messages("alice", self.a), [])
        self.assertEqual(len(self.memory.get_messages("alice", self.a)), 2)
        self.assertEqual(len(self.memory.get_rag_messages("bob", self.b)), 1)

    def test_invalid_details_never_save_half_turn(self):
        with self.assertRaises(ValueError):
            self.memory.append_turn("alice", self.a, "问题", "答案", details={"score": float("nan")})
        self.assertEqual(self.memory.get_messages("alice", self.a), [])
        with self.assertRaises(TypeError):
            self.memory.append_rag_message("alice", self.a, {"invalid": object()})
        self.assertEqual(self.memory.get_rag_messages("alice", self.a), [])


if __name__ == "__main__":
    unittest.main()
