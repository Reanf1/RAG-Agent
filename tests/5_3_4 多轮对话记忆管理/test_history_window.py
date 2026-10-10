"""5.3.4 多轮对话记忆管理：TestHistoryTokenWindow。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from copy import deepcopy
import json
import tempfile
import unittest
from unittest.mock import patch
from src.agent.memory import MemoryManager, count_history_tokens
from src.utils.config import load_config
from tests.helpers import isolated_agent_logs


def setUpModule():
    """保持既有 Agent 测试的模块级日志隔离。"""
    global _log_context
    _log_context = isolated_agent_logs()
    _log_context.__enter__()


def tearDownModule():
    _log_context.__exit__(None, None, None)


class TestHistoryTokenWindow(unittest.TestCase):
    """真实Qwen词表与SQLite窗口验证；历史样例明确构造，模型HTTP隔离。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.memory = MemoryManager(Path(self.directory.name) / "memory.sqlite3")
        self.session = self.memory.create_session("alice")
        self.config = deepcopy(load_config())
        self.config["memory"]["summary_trigger_turns"] = 1000  # 此组仅测试截断，摘要另组核验。
        patcher = patch("src.agent.memory.load_config", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)

    def add_turns(self, count=5):
        for index in range(count):
            self.memory.append_turn("alice", self.session, f"第{index}轮中文问题Transformer🙂", f"第{index}轮回答α² [1](论文第3页)")
        return self.history()

    def history(self):
        return [{"role": m.type, "content": m.content} for m in self.memory.get_messages("alice", self.session)]

    def test_token_count_matches_real_qwen_json_not_character_count(self):
        history = [{"role": "human", "content": "中文Transformer α²🙂"}, {"role": "ai", "content": "$x_1$ 引用[1]"}]
        self.assertEqual(count_history_tokens(history), 39)  # 固定官方词表的实际Token数量。
        self.assertNotEqual(count_history_tokens(history), len(json.dumps(history, ensure_ascii=False)))
        self.assertGreater(count_history_tokens(history), sum(len(item["content"]) for item in history))


    def test_over_limit_keeps_contiguous_recent_pairs_and_recounts_exact_suffix(self):
        history = self.add_turns(12)
        self.config["memory"]["max_history_tokens"] = count_history_tokens(history[-4:])
        context = self.memory.get_context("alice", self.session)
        self.assertEqual(context["history"], history[-4:])
        self.assertEqual(context["history_window"]["dropped_turns"], 10)
        self.assertEqual([m["role"] for m in context["history"]], ["human", "ai"] * 2)
        self.assertLessEqual(context["history_window"]["tokens"], context["history_window"]["max_tokens"])
        self.assertEqual(self.history(), history)


if __name__ == "__main__":
    unittest.main()
