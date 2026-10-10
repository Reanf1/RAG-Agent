"""5.3.4 多轮对话记忆管理：TestConversationSummary。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from copy import deepcopy
from io import BytesIO
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from src.agent.memory import MemoryManager
from src.utils.config import load_config
from tests.helpers import isolated_agent_logs


def setUpModule():
    """保持既有 Agent 测试的模块级日志隔离。"""
    global _log_context
    _log_context = isolated_agent_logs()
    _log_context.__enter__()


def tearDownModule():
    _log_context.__exit__(None, None, None)


class TestConversationSummary(unittest.TestCase):
    """真实SQLite/词表与构造HTTP响应：验证摘要替换、预算和失败不丢历史。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.memory = MemoryManager(Path(self.directory.name) / "memory.sqlite3")
        self.session = self.memory.create_session("alice")
        self.config = deepcopy(load_config())
        self.config["memory"].update(summary_trigger_turns=4, summary_keep_recent_turns=2, summary_max_tokens=120)
        patcher = patch("src.agent.memory.load_config", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def packet(summary="用户研究Transformer；当前实验代号EXP_FINAL，旧代号已纠正；引用paper.pdf第3页。", **overrides):
        packet = {"model": "qwen2.5:7b", "message": {"content": json.dumps({"summary": summary}, ensure_ascii=False)},
                  "done": True, "done_reason": "stop", "prompt_eval_count": 200, "eval_count": 40, **overrides}
        return BytesIO(json.dumps(packet, ensure_ascii=False).encode())

    def add(self, count=4, start=0):
        for index in range(start, start + count):
            self.memory.append_turn("alice", self.session, f"问题{index}：实验EXP_FINAL", f"回答{index}：[1]paper.pdf第3页")
        return [{"role": m.type, "content": m.content} for m in self.memory.get_messages("alice", self.session)]

    def saved(self):
        with sqlite3.connect(self.memory.db_path) as connection:
            return connection.execute("SELECT content, through_message_id FROM summaries WHERE session_id=?", (self.session,)).fetchone()

    def compress(self, **kwargs):
        with patch("src.agent.react_loop.urlopen", return_value=self.packet(**kwargs)) as http:
            return self.memory.get_context("alice", self.session), http


    def test_threshold_replaces_only_old_pairs_and_keeps_full_archive(self):
        history = self.add()
        context, http = self.compress()
        self.assertEqual(context["history"], history[-4:])
        self.assertIn("EXP_FINAL", context["summary"])
        self.assertEqual(context["memory_summary"]["summarized_turns"], 2)
        self.assertEqual(context["memory_summary"]["unsummarized_dropped_turns"], 0)
        self.assertEqual(context["memory_summary"]["calls"][0]["usage"], {"prompt_eval_count": 200, "eval_count": 40})
        self.assertTrue(context["memory_summary"]["calls"][0]["saved"])
        source = json.loads(json.loads(http.call_args.args[0].data)["messages"][1]["content"])
        self.assertEqual(source, {"previous_summary": "", "older_history": history[:4]})
        self.assertEqual(len(self.memory.get_messages("alice", self.session)), 8)


    def test_model_timeout_preserves_previous_summary_and_retries_next_request(self):
        self.add()
        initial, _ = self.compress()
        saved = self.saved()
        self.add(2, start=4)
        with patch("src.agent.react_loop.urlopen", side_effect=TimeoutError("明确注入的超时")):
            failed = self.memory.get_context("alice", self.session)
        self.assertEqual(failed["summary"], initial["summary"])
        self.assertEqual(self.saved(), saved)
        self.assertIn("超时", failed["memory_summary"]["warning"])
        retried, _ = self.compress(summary="用户确认EXP_NEXT。")
        self.assertEqual(retried["memory_summary"]["summarized_turns"], 4)
        self.assertEqual(len(self.memory.get_messages("alice", self.session)), 12)


if __name__ == "__main__":
    unittest.main()
