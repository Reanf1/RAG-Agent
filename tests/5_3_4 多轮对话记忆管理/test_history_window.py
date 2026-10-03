"""5.3.4 多轮对话记忆管理：TestHistoryTokenWindow。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from copy import deepcopy
from io import BytesIO
import json
import tempfile
import unittest
from unittest.mock import patch
from src.agent.memory import MemoryManager, count_history_tokens, run_session
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

    def test_empty_history_uses_zero_budget_even_with_minimum_limit(self):
        self.config["memory"]["max_history_tokens"] = 1
        context = self.memory.get_context("alice", self.session)
        self.assertEqual(context["history"], [])
        self.assertEqual(context["history_window"]["tokens"], 0)
        self.assertEqual(context["history_window"]["dropped_turns"], 0)

    def test_under_budget_keeps_original_roles_content_order_and_archive(self):
        history = self.add_turns(3)
        context = self.memory.get_context("alice", self.session)
        self.assertEqual(context["history"], history)
        self.assertEqual(context["history_window"]["retained_turns"], 3)
        self.assertEqual(context["history_window"]["dropped_turns"], 0)
        self.assertEqual(context["history_window"]["tokens"], count_history_tokens(history))
        self.assertEqual(self.history(), history)

    def test_exact_limit_keeps_all_and_one_token_less_removes_complete_oldest_turn(self):
        history = self.add_turns(3)
        self.config["memory"]["max_history_tokens"] = count_history_tokens(history)
        self.assertEqual(self.memory.get_context("alice", self.session)["history"], history)
        self.config["memory"]["max_history_tokens"] -= 1
        context = self.memory.get_context("alice", self.session)
        self.assertEqual(context["history"], history[2:])
        self.assertEqual(context["history_window"]["dropped_turns"], 1)

    def test_over_limit_keeps_contiguous_recent_pairs_and_recounts_exact_suffix(self):
        history = self.add_turns(12)
        self.config["memory"]["max_history_tokens"] = count_history_tokens(history[-4:])
        context = self.memory.get_context("alice", self.session)
        self.assertEqual(context["history"], history[-4:])
        self.assertEqual(context["history_window"]["dropped_turns"], 10)
        self.assertEqual([m["role"] for m in context["history"]], ["human", "ai"] * 2)
        self.assertLessEqual(context["history_window"]["tokens"], context["history_window"]["max_tokens"])
        self.assertEqual(self.history(), history)

    def test_one_oversized_latest_turn_leaves_empty_window_without_orphan_answer(self):
        self.add_turns(1)
        self.memory.append_turn("alice", self.session, "超长中文论文问题" * 500, "超长回答" * 500)
        self.config["memory"]["max_history_tokens"] = 100
        context = self.memory.get_context("alice", self.session)
        self.assertEqual(context["history"], [])
        self.assertEqual(context["history_window"]["dropped_turns"], 2)
        self.assertEqual(context["history_window"]["tokens"], 0)
        self.assertEqual(len(self.history()), 4)

    def test_oversized_old_turn_is_removed_and_recent_formula_citation_unchanged(self):
        self.memory.append_turn("alice", self.session, "旧论文" * 500, "旧回答" * 500)
        self.memory.append_turn("alice", self.session, "What is α²🙂?", "答案 $x_1$，来源[1](paper.pdf第3页)。")
        history = self.history()
        self.config["memory"]["max_history_tokens"] = count_history_tokens(history[-2:])
        self.assertEqual(self.memory.get_context("alice", self.session)["history"], history[-2:])

    def test_reopened_manager_and_changed_budget_recompute_window_without_losing_archive(self):
        history = self.add_turns(5)
        self.config["memory"]["max_history_tokens"] = count_history_tokens(history[-2:])
        self.assertEqual(self.memory.get_context("alice", self.session)["history"], history[-2:])
        self.config["memory"]["max_history_tokens"] = count_history_tokens(history)
        reopened = MemoryManager(self.memory.db_path)
        self.assertEqual(reopened.get_context("alice", self.session)["history"], history)

    def test_window_isolated_from_same_user_other_session_and_other_user(self):
        other = self.memory.create_session("alice")
        bob = self.memory.create_session("bob")
        self.add_turns(8)
        self.memory.append_turn("alice", other, "ALICE_OTHER", "ALICE_OTHER_ANSWER")
        self.memory.append_turn("bob", bob, "BOB_PRIVATE", "BOB_PRIVATE_ANSWER")
        self.config["memory"]["max_history_tokens"] = 100
        self.assertNotIn("PRIVATE", json.dumps(self.memory.get_context("alice", self.session)))
        self.assertEqual(self.memory.get_context("alice", other)["history_window"]["dropped_turns"], 0)
        self.assertEqual(self.memory.get_context("bob", bob)["history_window"]["dropped_turns"], 0)
        with patch("src.agent.memory.count_history_tokens") as counter, self.assertRaises(PermissionError):
            self.memory.get_context("bob", self.session)
        counter.assert_not_called()
        self.assertEqual(len(self.memory.get_messages("bob", bob)), 2)

    def test_invalid_limits_refused_before_model_or_archive_changes(self):
        history = self.add_turns(1)
        for limit in (0, -1, True, 1.5, "2000", None):
            self.config["memory"]["max_history_tokens"] = limit
            with self.subTest(limit=limit), patch("src.agent.react_loop.run_react") as run, self.assertRaises(ValueError):
                list(run_session("问题", "alice", self.session, memory=self.memory))
            run.assert_not_called()
            self.assertEqual(self.history(), history)

    def test_missing_tokenizer_refuses_model_without_network_or_character_fallback(self):
        self.add_turns(1)
        self.config["memory"]["tokenizer_path"] = str(Path(self.directory.name) / "missing.json")
        with patch("src.agent.react_loop.run_react") as run, patch("socket.create_connection") as network, self.assertRaises(FileNotFoundError):
            list(run_session("问题", "alice", self.session, memory=self.memory))
        run.assert_not_called()
        network.assert_not_called()
        self.assertEqual(len(self.history()), 2)

    def test_other_model_cannot_silently_use_qwen_tokenizer(self):
        self.add_turns(1)
        self.config["llm"]["model"] = "chatglm:latest"
        with self.assertRaisesRegex(ValueError, "对应分词器"):
            self.memory.get_context("alice", self.session)

    def test_runtime_count_needs_no_network_or_llm(self):
        history = self.add_turns(1)
        from src.agent.memory import _load_tokenizer
        _load_tokenizer.cache_clear()  # 强制从本地词表重新读取，验证并非仅缓存命中。
        with patch("socket.create_connection", side_effect=AssertionError("不能联网")), patch("src.agent.react_loop.urlopen") as llm:
            context = self.memory.get_context("alice", self.session)
        llm.assert_not_called()
        self.assertEqual(context["history_window"]["tokens"], count_history_tokens(history))

    def test_agent_sends_only_window_in_every_actual_request_and_keeps_original_archive(self):
        self.memory.append_turn("alice", self.session, "OUTSIDE_WINDOW_OLD " * 300, "不再送入模型的旧回答" * 300)
        self.memory.append_turn("alice", self.session, "最新实验代号为EXP_RECENT。", "已记录EXP_RECENT。")
        self.config["memory"]["max_history_tokens"] = 120
        expected = self.memory.get_context("alice", self.session)
        plan = {"thought": "根据可见最近历史回答。", "next_step": "answer", "tool_name": None}
        answer = {"observation": "最近历史提供了代号。", "decision": "finish", "task_complete": True, "answer": "EXP_RECENT"}
        def packet(content):
            return BytesIO(json.dumps({"model": "qwen2.5:7b", "message": {"content": json.dumps(content)},
                        "done": True, "done_reason": "stop", "prompt_eval_count": 300, "eval_count": 30}).encode())
        with patch("src.agent.react_loop.urlopen", side_effect=[packet(plan), packet(answer)]) as http:
            events = list(run_session("最近的代号是什么？", "alice", self.session, tools=[], memory=self.memory))
        self.assertTrue(events[-1]["task_complete"])
        self.assertEqual(events[-1]["context"]["history_window"], expected["history_window"])
        self.assertEqual(http.call_count, 2)
        for call in http.call_args_list:
            body = call.args[0].data.decode()
            self.assertNotIn("OUTSIDE_WINDOW_OLD", body)
            self.assertIn("EXP_RECENT", body)
            context = json.loads(json.loads(body)["messages"][1]["content"])["context"]
            self.assertEqual(context["history"], expected["history"])
            self.assertLessEqual(count_history_tokens(context["history"]), 120)
        self.assertEqual(len(self.history()), 6)


if __name__ == "__main__":
    unittest.main()
