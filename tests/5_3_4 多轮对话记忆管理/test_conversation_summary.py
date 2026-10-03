"""5.3.4 多轮对话记忆管理：TestConversationSummary。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
import json
import sqlite3
import tempfile
from threading import Barrier
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

    def test_below_threshold_no_summary_or_model_call(self):
        history = self.add(3)
        with patch("src.agent.react_loop.urlopen") as http:
            context = self.memory.get_context("alice", self.session)
        http.assert_not_called()
        self.assertEqual(context["history"], history)
        self.assertNotIn("summary", context)
        self.assertIsNone(self.saved())

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

    def test_native_prompt_schema_local_endpoint_and_input_output_limits(self):
        self.add()
        _, http = self.compress()
        request = http.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(body["format"]["required"], ["summary"])
        self.assertFalse(body["stream"])
        self.assertEqual(body["options"]["num_predict"], 248)
        self.assertEqual(http.call_args.kwargs["timeout"], 60)
        self.assertIn("最新纠正", body["messages"][0]["content"])

    def test_persistent_summary_reused_without_repeated_model_call(self):
        self.add()
        original, _ = self.compress()
        with patch("src.agent.react_loop.urlopen") as http:
            reopened = MemoryManager(self.memory.db_path).get_context("alice", self.session)
        http.assert_not_called()
        self.assertEqual(reopened["summary"], original["summary"])
        self.assertEqual(reopened["history"], original["history"])
        self.assertEqual(reopened["memory_summary"]["calls"], [])

    def test_incremental_compression_merges_previous_summary_only_with_new_old_pairs(self):
        history = self.add()
        first, _ = self.compress()
        self.add(2, start=4)
        second, http = self.compress(summary="EXP_FINAL已更新为EXP_NEXT；保留paper.pdf第3页与待办。")
        source = json.loads(json.loads(http.call_args.args[0].data)["messages"][1]["content"])
        self.assertEqual(source["previous_summary"], first["summary"])
        self.assertEqual(source["older_history"], history[4:])
        self.assertEqual(second["memory_summary"]["summarized_turns"], 4)
        self.assertEqual(len(second["history"]), 4)
        self.assertGreater(self.saved()[1], 4)

    def test_summary_and_recent_history_share_real_token_budget(self):
        from src.agent.memory import count_memory_tokens
        self.add()
        initial, _ = self.compress()
        self.config["memory"]["max_history_tokens"] = count_memory_tokens(initial["history"][-2:], initial["summary"])
        result = self.memory.get_context("alice", self.session)
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(result["history_window"]["tokens"], self.config["memory"]["max_history_tokens"])
        self.assertEqual(result["memory_summary"]["unsummarized_dropped_turns"], 1)

    def test_smaller_budget_omits_whole_summary_without_destroying_saved_text(self):
        self.add()
        original, _ = self.compress()
        self.config["memory"]["max_history_tokens"] = 1
        context = self.memory.get_context("alice", self.session)
        self.assertEqual(context["history"], [])
        self.assertNotIn("summary", context)
        self.assertLessEqual(context["history_window"]["tokens"], 1)
        self.assertIn("本轮省略", context["memory_summary"]["warning"])
        self.config["memory"]["max_history_tokens"] = 2000
        self.assertEqual(self.memory.get_context("alice", self.session)["summary"], original["summary"])

    def test_invalid_and_partial_responses_keep_archive_and_do_not_commit(self):
        history = self.add()
        packets = [self.packet(done=False), self.packet(done_reason="length"), self.packet(error="模型故障"),
                   self.packet(model=""), self.packet(message={}), self.packet(summary=" "),
                   self.packet(message={"content": "{坏JSON"}), self.packet(message={"content": '{"summary":"摘要","extra":1}'}),
                   self.packet(summary="超长摘要" * 500)]
        for packet in packets:
            with self.subTest(packet=packet), patch("src.agent.react_loop.urlopen", return_value=packet):
                context = self.memory.get_context("alice", self.session)
            self.assertNotIn("summary", context)
            self.assertIn("未完成", context["memory_summary"]["warning"])
            self.assertIsNone(self.saved())
            self.assertEqual(context["history"], history)

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

    def test_summary_database_failure_rolls_back_without_losing_raw_history(self):
        self.add()
        with sqlite3.connect(self.memory.db_path) as connection:
            connection.executescript("""CREATE TRIGGER fail_summary BEFORE INSERT ON summaries
                BEGIN SELECT RAISE(ABORT, '明确注入摘要入库故障'); END;""")
        context, _ = self.compress()
        self.assertIsNone(self.saved())
        self.assertFalse(context["memory_summary"]["calls"][0]["saved"])
        self.assertIn("入库故障", context["memory_summary"]["warning"])
        self.assertEqual(len(self.memory.get_messages("alice", self.session)), 8)

    def test_owned_clear_removes_summary_and_history_but_other_session_stays(self):
        self.add()
        self.compress()
        other = self.memory.create_session("bob")
        self.memory.append_turn("bob", other, "BOB_PRIVATE", "回答")
        with self.assertRaises(PermissionError):
            self.memory.clear_session("bob", self.session)
        self.assertIsNotNone(self.saved())
        self.memory.clear_session("alice", self.session)
        self.assertIsNone(self.saved())
        self.assertEqual(self.memory.get_context("alice", self.session)["history"], [])
        self.assertEqual(len(self.memory.get_messages("bob", other)), 2)

    def test_summary_request_contains_no_other_user_or_same_user_other_session(self):
        self.add()
        for user in ("alice", "bob"):
            other = self.memory.create_session(user)
            self.memory.append_turn(user, other, f"{user}_PRIVATE", "不能混入摘要")
        _, http = self.compress()
        self.assertNotIn("PRIVATE", http.call_args.args[0].data.decode())
        with patch("src.agent.react_loop.urlopen") as denied, self.assertRaises(PermissionError):
            self.memory.get_context("bob", self.session)
        denied.assert_not_called()

    def test_clear_during_generation_discards_stale_summary_and_returns_empty_context(self):
        self.add()
        def clear_and_reply(*args, **kwargs):
            self.memory.clear_session("alice", self.session)
            return self.packet()
        with patch("src.agent.react_loop.urlopen", side_effect=clear_and_reply):
            context = self.memory.get_context("alice", self.session)
        self.assertIsNone(self.saved())
        self.assertEqual(context["history"], [])
        self.assertNotIn("summary", context)
        self.assertIn("过期结果", context["memory_summary"]["warning"])

    def test_simultaneous_summaries_cannot_overwrite_same_boundary(self):
        self.add()
        barrier = Barrier(2)
        def reply(*args, **kwargs):
            barrier.wait(timeout=5)
            return self.packet()
        with patch("src.agent.react_loop.urlopen", side_effect=reply), ThreadPoolExecutor(max_workers=2) as pool:
            contexts = list(pool.map(lambda _: self.memory.get_context("alice", self.session), range(2)))
        self.assertEqual(sum(c["memory_summary"]["calls"][0]["saved"] for c in contexts), 1)
        self.assertTrue(any("过期结果" in c["memory_summary"]["warning"] for c in contexts))
        self.assertEqual(self.saved()[1], 4)

    def test_oversized_single_source_is_not_silently_truncated_or_marked_compressed(self):
        self.memory.append_turn("alice", self.session, "不可截断的旧问题" * 2000, "旧回答" * 2000)
        self.add(3, start=1)
        with patch("src.agent.react_loop.urlopen") as http:
            context = self.memory.get_context("alice", self.session)
        http.assert_not_called()
        self.assertIsNone(self.saved())
        self.assertIn("单轮超过", context["memory_summary"]["warning"])
        self.assertEqual(len(self.memory.get_messages("alice", self.session)), 8)

    def test_large_archive_uses_bounded_complete_pairs_and_three_call_limit(self):
        from src.agent.memory import count_memory_tokens
        for index in range(7):
            self.memory.append_turn("alice", self.session, f"问题{index}" + "word " * 1800, "答案")
        with patch("src.agent.react_loop.urlopen", side_effect=lambda *a, **k: self.packet()) as http:
            context = self.memory.get_context("alice", self.session)
        self.assertEqual(http.call_count, 3)
        self.assertEqual(context["memory_summary"]["summarized_turns"], 3)
        self.assertIn("三批", context["memory_summary"]["warning"])
        for call in http.call_args_list:
            messages = json.loads(call.args[0].data)["messages"]
            self.assertLessEqual(count_history_tokens(messages), self.config["llm"]["num_ctx"] // 2)
            self.assertEqual(len(json.loads(messages[1]["content"])["older_history"]), 2)
        self.assertLessEqual(count_memory_tokens(context["history"], context["summary"]), 2000)

    def test_invalid_summary_settings_refused_before_model_call(self):
        self.add()
        for key, value in (("summary_trigger_turns", 2), ("summary_keep_recent_turns", 0),
                           ("summary_max_tokens", True), ("summary_trigger_turns", "10")):
            original = self.config["memory"][key]
            self.config["memory"][key] = value
            with self.subTest(key=key, value=value), patch("src.agent.react_loop.urlopen") as http, self.assertRaises(ValueError):
                self.memory.get_context("alice", self.session)
            http.assert_not_called()
            self.config["memory"][key] = original

    def test_agent_receives_summary_as_data_in_all_stages_not_old_original_history(self):
        self.add()
        self.compress()
        plan = {"thought": "使用本会话摘要回答。", "next_step": "answer", "tool_name": None}
        answer = {"observation": "摘要记录了代号。", "decision": "finish", "task_complete": True, "answer": "EXP_FINAL"}
        def packet(content):
            return BytesIO(json.dumps({"model": "qwen2.5:7b", "message": {"content": json.dumps(content)},
                                      "done": True, "done_reason": "stop"}).encode())
        with patch("src.agent.react_loop.urlopen", side_effect=[packet(plan), packet(answer)]) as http:
            events = list(run_session("我的代号是什么？", "alice", self.session, tools=[], memory=self.memory))
        self.assertTrue(events[-1]["task_complete"])
        self.assertEqual(len(self.memory.get_messages("alice", self.session)), 10)
        for call in http.call_args_list:
            payload = json.loads(call.args[0].data)
            context = json.loads(payload["messages"][1]["content"])["context"]
            self.assertIn("EXP_FINAL", context["summary"])
            self.assertEqual([m["role"] for m in context["history"]], ["human", "ai"] * 2)
            self.assertNotIn("问题0", json.dumps(context, ensure_ascii=False))
            self.assertNotIn(context["summary"], payload["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
