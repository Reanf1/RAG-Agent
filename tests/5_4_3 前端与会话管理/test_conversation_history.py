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

    def test_deleted_session_rejects_reads_writes_and_generation(self):
        self.memory.delete_session("alice", self.a)
        for operation in (lambda: self.memory.get_messages("alice", self.a),
                          lambda: self.memory.append_turn("alice", self.a, "问", "答"),
                          lambda: list(run_session("问", "alice", self.a, memory=self.memory))):
            with self.subTest(operation=operation), patch("src.agent.react_loop.run_react") as model:
                with self.assertRaises(LookupError):
                    operation()
                model.assert_not_called()
        # 会话已删除，其他用户再删除时按"不存在"拒绝。
        with self.assertRaises(LookupError):
            self.memory.delete_session("bob", self.a)
        self.assertEqual(self.memory.list_sessions("alice"), [])

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

    def test_invalid_details_never_save_half_turn(self):
        with self.assertRaises(ValueError):
            self.memory.append_turn("alice", self.a, "问题", "答案", details={"score": float("nan")})
        self.assertEqual(self.memory.get_messages("alice", self.a), [])


if __name__ == "__main__":
    unittest.main()
