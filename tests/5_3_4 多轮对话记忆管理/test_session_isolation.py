"""5.3.4 多轮对话记忆管理：TestSessionIsolation。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from io import BytesIO
import json
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


class TestSessionIsolation(unittest.TestCase):
    """真实临时SQLite隔离测试；模型HTTP按既有方式隔离，不替换存储和Agent循环。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "sessions" / "memory.sqlite3"
        self.memory = MemoryManager(self.path)
        self.a1 = self.memory.create_session("alice")
        self.a2 = self.memory.create_session("alice")
        self.b1 = self.memory.create_session("bob")

    def contents(self, user, session):
        return [message.content for message in self.memory.get_messages(user, session)]

    def packet(self, content):
        return {"model": "qwen2.5:7b", "message": {"content": json.dumps(content, ensure_ascii=False)},
                "done": True, "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 20}

    def packets(self, answer="回答", complete=True):
        return [self.packet({"thought": "结合本会话已有历史回答。", "next_step": "answer", "tool_name": None}),
                self.packet({"observation": "已有信息足够回答。", "decision": "finish",
                             "task_complete": complete, "answer": answer})]

    def run_turn(self, user, session, question="追问", answer="回答", complete=True):
        with patch("src.agent.react_loop.urlopen", side_effect=[BytesIO(json.dumps(p).encode()) for p in self.packets(answer, complete)]) as http:
            events = list(run_session(question, user, session, tools=[], memory=self.memory))
        return events, http


    def test_same_user_different_sessions_have_independent_history(self):
        self.memory.append_turn("alice", self.a1, "A的问题", "A的回答")
        self.memory.append_turn("alice", self.a2, "B的问题", "B的回答")
        self.assertEqual(self.contents("alice", self.a1), ["A的问题", "A的回答"])
        self.assertEqual(self.contents("alice", self.a2), ["B的问题", "B的回答"])


    def test_foreign_session_read_append_and_clear_are_rejected_without_changes(self):
        self.memory.append_turn("alice", self.a1, "私有问题", "私有回答")
        for operation in (lambda: self.memory.get_messages("bob", self.a1),
                          lambda: self.memory.append_turn("bob", self.a1, "覆盖", "覆盖"),
                          lambda: self.memory.clear_session("bob", self.a1)):
            with self.assertRaises(PermissionError):
                operation()
        self.assertEqual(self.contents("alice", self.a1), ["私有问题", "私有回答"])
        self.assertEqual(self.contents("bob", self.b1), [])


    def test_reopening_manager_restores_both_users_without_shared_message_cache(self):
        self.memory.append_turn("alice", self.a1, "持久问题A", "持久回答A")
        self.memory.append_turn("bob", self.b1, "持久问题B", "持久回答B")
        reopened = MemoryManager(self.path)
        self.assertEqual(reopened.get_messages("alice", self.a1)[0].content, "持久问题A")
        self.assertEqual(reopened.get_messages("bob", self.b1)[0].content, "持久问题B")
        reopened.append_turn("alice", self.a1, "追加", "答复")
        self.assertEqual(len(self.memory.get_messages("alice", self.a1)), 4)


if __name__ == "__main__":
    unittest.main()
