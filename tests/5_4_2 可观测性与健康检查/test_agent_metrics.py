"""5.4.2 可观测性与健康检查：TestAgentMetrics、TestAgentTraceMetrics、TestAgentMetricsEntryAndPage。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from copy import deepcopy
import json
import sqlite3
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from src.agent.memory import MemoryManager, run_session
from src.agent.react_loop import run_react
from src.agent.router import execute_calls
from src.utils.config import load_config
from tests.helpers import isolated_agent_logs


def setUpModule():
    """保持既有 Agent 测试的模块级日志隔离。"""
    global _log_context
    _log_context = isolated_agent_logs()
    _log_context.__enter__()


def tearDownModule():
    _log_context.__exit__(None, None, None)


class TestAgentMetrics(unittest.TestCase):
    """使用明确用量的事件验证记账，不把模拟Token当作真实性能数据。"""

    @staticmethod
    def events():
        native = AIMessage(content="", tool_calls=[{"id": "a", "name": "calculator", "args": {}},
                                                   {"id": "b", "name": "current_time", "args": {}}])
        return [{"type": "thought", "iteration": 1, "usage": {"prompt_eval_count": 10, "eval_count": 2}},
                {"type": "tool_call", "iteration": 1, "name": "calculator", "call_id": "a", "message": native,
                 "usage": {"prompt_eval_count": 100, "eval_count": 20}},
                {"type": "tool_call", "iteration": 1, "name": "current_time", "call_id": "b",
                 "usage": {"prompt_eval_count": 0, "eval_count": 0}},
                {"type": "tool_result", "iteration": 1, "name": "calculator", "call_id": "a", "status": "success",
                 "result": {"usage": {"prompt_eval_count": 5, "eval_count": 3}}},
                {"type": "tool_result", "iteration": 1, "name": "current_time", "call_id": "b", "status": "success",
                 "result": {"usage": {"prompt_eval_count": 7, "eval_count": 4}}},
                {"type": "observation", "iteration": 1, "usage": {"prompt_eval_count": 8, "eval_count": 2}},
                {"type": "done", "iterations": 1, "task_complete": True, "stop_reason": "task_complete", "full_response": "结果为6。"}]

    def collect(self, events):
        from src.utils.logger import update_agent_metrics
        result = {}
        for event in events:
            result = update_agent_metrics(result, event)
        return result

    def test_parallel_action_only_once_and_tool_usage_separate(self):
        result = self.collect(self.events())
        self.assertEqual(result["tokens"]["total"], 161)
        self.assertEqual(len(result["calls"]), 5)
        self.assertEqual(result["calls"][1]["phase"], "Action（共享）")
        self.assertEqual(result["calls"][1]["tool"], "calculator + current_time")
        self.assertEqual(result["calls"][1]["tool_call_ids"], ["a", "b"])
        self.assertEqual([c["id"] for c in result["calls"][2:4]], ["a", "b"])

    def test_missing_usage_preserves_known_part_and_unknown_total(self):
        events = self.events()
        del events[3]["result"]["usage"]["eval_count"]
        tokens = self.collect(events)["tokens"]
        self.assertIsNone(tokens["total"])
        self.assertEqual(tokens["known_total"], 158)
        self.assertEqual(tokens["unknown_calls"], 1)

    def test_invalid_usage_is_not_zero_or_negative_consumption(self):
        tokens = self.collect([{"type": "thought", "usage": {"prompt_eval_count": True, "eval_count": -2}}])["tokens"]
        self.assertIsNone(tokens["total"])
        self.assertEqual(tokens["known_total"], 0)

    def test_retry_does_not_claim_only_final_usage_as_total(self):
        event = self.events()[3]
        event["attempts"] = [{"status": "error"}, {"status": "success"}]
        result = self.collect([event])
        self.assertIsNone(result["tokens"]["total"])
        self.assertEqual(result["tokens"]["known_total"], 8)
        self.assertTrue(result["calls"][0]["incomplete"])

    def test_failure_and_timeout_cannot_report_zero_tokens(self):
        for event in ({"type": "error"}, {**self.events()[3], "status": "error", "result": None}):
            with self.subTest(event=event):
                self.assertIsNone(self.collect([event])["tokens"]["total"])

    def test_rule_and_non_model_tool_can_report_true_zero(self):
        events = [{"type": "thought", "usage": {"prompt_eval_count": 0, "eval_count": 0}},
                  {**self.events()[3], "result": {"usage": {"prompt_eval_count": 0, "eval_count": 0}}}]
        self.assertEqual(self.collect(events)["tokens"]["total"], 0)

    def test_snapshot_does_not_mutate_previous_event_or_input(self):
        from src.utils.logger import update_agent_metrics
        before = self.collect(self.events()[:1])
        original = deepcopy(before)
        after = update_agent_metrics(before, self.events()[1])
        self.assertEqual(before, original)
        self.assertEqual(after["tokens"]["total"], 132)

    def test_rag_result_records_returned_count_and_real_retrieval_time(self):
        event = {**self.events()[3], "name": "knowledge_base_search", "result": {
            "usage": {"prompt_eval_count": 0, "eval_count": 0}, "retrieval_seconds": 0.42,
            "retrieval": {"request_id": "rag1", "status": "empty", "returned_chunks": 0}}}
        self.assertEqual(self.collect([event])["retrievals"], [{"call_id": "a", "request_id": "rag1",
            "status": "empty", "returned_chunks": 0, "seconds": 0.42}])

    def test_memory_summary_only_counts_current_calls_and_request_ids_change(self):
        summary = {"memory_summary": {"calls": [{"usage": {"prompt_eval_count": 30, "eval_count": 4}}]}}
        with patch("src.agent.react_loop._run_react", side_effect=lambda *args: iter(deepcopy(self.events()))):
            first = list(run_react("问题", context=summary))
            second = list(run_react("后续问题", context={"memory_summary": {"calls": []}}))
        self.assertEqual(first[-1]["metrics"]["tokens"]["total"], 195)
        self.assertEqual(second[-1]["metrics"]["tokens"]["total"], 161)
        self.assertNotEqual(first[-1]["request_id"], second[-1]["request_id"])
        self.assertEqual(first[0]["metrics"]["tokens"]["total"], 46)
        self.assertTrue(all(e["request_id"] == first[0]["request_id"] for e in first))

    def test_retrieval_hit_rate_excludes_cache_not_called_and_failure(self):
        from src.utils.logger import retrieval_request_metrics
        records = [{"retrieval": {"status": status, "documents": docs}, "cache": {"hit": cache},
                    "timing": {"retrieval_seconds": seconds, "response_seconds": seconds * 2}}
                   for status, docs, cache, seconds in [("success", [1], False, 1), ("empty", [], False, 2),
                       ("error", [], False, 3), ("success", [1], True, 0), ("not_started", [], False, 0)]]
        with patch("src.utils.logger.read_rag_requests", return_value=(records, 2)):
            result = retrieval_request_metrics()
        self.assertEqual((result["attempts"], result["completed"], result["hits"], result["failed"]), (3, 2, 1, 1))
        self.assertEqual(result["hit_rate"], 0.5)
        self.assertEqual((result["retrieval_seconds"], result["response_seconds"]), (2, 4))
        self.assertEqual(result["invalid_lines"], 2)

    def test_no_retrieval_samples_do_not_show_zero_percent(self):
        from src.utils.logger import retrieval_request_metrics
        with patch("src.utils.logger.read_rag_requests", return_value=([], 0)):
            result = retrieval_request_metrics()
        self.assertIsNone(result["hit_rate"])
        self.assertIsNone(result["retrieval_seconds"])

    def test_pending_generation_is_not_a_finished_response_latency(self):
        from src.utils.logger import retrieval_request_metrics
        records = [{"status": "retrieved", "retrieval": {"status": "success", "documents": [1]},
                    "timing": {"retrieval_seconds": 0.3, "response_seconds": 0.3}}]
        with patch("src.utils.logger.read_rag_requests", return_value=(records, 0)):
            result = retrieval_request_metrics()
        self.assertEqual(result["hit_rate"], 1)
        self.assertEqual(result["retrieval_seconds"], 0.3)
        self.assertIsNone(result["response_seconds"])


class TestAgentTraceMetrics(unittest.TestCase):
    """核验轨迹事实、逻辑调用计数和真实线程执行；模拟事件不作为性能结论。"""

    def collect(self, events):
        from src.utils.logger import update_agent_metrics
        result = {}
        for event in events:
            result = update_agent_metrics(result, event)
        return result

    def test_trace_keeps_all_parallel_actions_and_public_fields_only(self):
        events = TestAgentMetrics.events()
        events[0].update(thought="独立执行计算和时间查询。", reasoning_content="私有字段", context={"secret": "历史"})
        events[1]["args"] = {"expression": "2*3"}
        result = self.collect(events)
        self.assertEqual([t["type"] for t in result["trace"]], [e["type"] for e in events])
        self.assertEqual(result["trace"][-1]["iteration"], 1)
        self.assertEqual(result["trace"][1]["args"], {"expression": "2*3"})
        self.assertEqual(result["trace"][3]["result"], events[3]["result"])
        serialized = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("私有字段", serialized)
        self.assertNotIn("secret", serialized)
        self.assertTrue(all("message" not in t for t in result["trace"]))
        self.assertEqual(result["tokens"]["total"], 161)

    def test_parallel_pending_calls_do_not_enter_success_rate_denominator(self):
        events = TestAgentMetrics.events()
        pending = self.collect(events[:3])
        self.assertEqual((pending["tools"]["started"], pending["tools"]["pending"]), (2, 2))
        self.assertIsNone(pending["tools"]["success_rate"])
        partial = self.collect(events[:4])["tools"]
        self.assertEqual((partial["completed"], partial["pending"], partial["success_rate"]), (1, 1, 1))

    def test_same_tool_different_ids_and_duplicate_snapshot_count_once(self):
        events = TestAgentMetrics.events()
        events[2]["name"] = events[4]["name"] = "calculator"
        events[3]["elapsed_seconds"] = 0.2
        events[4].update(status="error", elapsed_seconds=0.6)
        result = self.collect(events + [events[4]])
        self.assertEqual(len(result["tool_calls"]), 2)
        self.assertEqual((result["tools"]["successes"], result["tools"]["failures"]), (1, 1))
        self.assertEqual(result["tools"]["success_rate"], 0.5)
        self.assertAlmostEqual(result["tools"]["mean_seconds"], 0.4)
        self.assertEqual(len(result["tools"]["by_tool"]), 1)
        self.assertEqual(result["tools"]["by_tool"][0]["success_rate"], 0.5)

    def test_retry_success_is_one_call_and_not_task_completion(self):
        event = TestAgentMetrics.events()[3]
        event.update(attempts=[{"status": "error"}, {"status": "success"}], elapsed_seconds=0.8)
        event["result"]["status"] = "needs_confirmation"
        result = self.collect([event, {"type": "done", "iterations": 1, "task_complete": False}])
        self.assertEqual(result["tools"]["success_rate"], 1)
        self.assertEqual(result["tools"]["completed"], 1)
        self.assertEqual(result["tool_calls"][0]["attempts"], 2)
        self.assertEqual(result["tools"]["mean_seconds"], 0.8)
        self.assertFalse(result["trace"][-1]["task_complete"])

    def test_deadline_failure_preserves_background_pending_fact(self):
        event = {**TestAgentMetrics.events()[3], "status": "error", "error_kind": "deadline",
                 "pending": True, "attempts": [{"attempt": 1, "status": "running"}], "elapsed_seconds": 0.1}
        result = self.collect([event])
        self.assertEqual((result["tools"]["completed"], result["tools"]["failures"]), (1, 1))
        self.assertEqual(result["tools"]["success_rate"], 0)
        self.assertTrue(result["trace"][0]["pending"])
        self.assertEqual(result["trace"][0]["attempts"][0]["status"], "running")

    def test_no_tools_and_skipped_action_keep_no_samples(self):
        result = self.collect([{"type": "thought", "iteration": 1, "thought": "直接回答。"},
            {"type": "action_skipped", "iteration": 1, "reason": "无需工具。"},
            {"type": "observation", "iteration": 1, "decision": "finish"},
            {"type": "done", "iterations": 1, "stop_reason": "task_complete"}])
        self.assertEqual(result["tools"]["started"], 0)
        self.assertIsNone(result["tools"]["success_rate"])
        self.assertIsNone(result["tools"]["mean_seconds"])
        self.assertEqual(result["trace"][1]["reason"], "无需工具。")

    def test_missing_or_invalid_time_is_not_reported_as_zero(self):
        for seconds in (None, -1, True, float("inf"), float("nan")):
            with self.subTest(seconds=seconds):
                result = self.collect([{**TestAgentMetrics.events()[3], "elapsed_seconds": seconds}])
                self.assertIsNone(result["tools"]["mean_seconds"])
                self.assertIsNone(result["tool_calls"][0]["seconds"])
                self.assertEqual(result["tools"]["success_rate"], 1)

    def test_snapshots_and_new_requests_do_not_share_trace(self):
        from src.utils.logger import update_agent_metrics
        event = TestAgentMetrics.events()[3]
        first = update_agent_metrics({}, event)
        second = update_agent_metrics(first, {"type": "recovery", "iteration": 1, "message": "尝试替代工具。"})
        event["result"]["usage"]["eval_count"] = 99
        self.assertEqual(len(first["trace"]), 1)
        self.assertEqual(first["trace"][0]["result"]["usage"]["eval_count"], 3)
        self.assertEqual(second["trace"][-1]["message"], "尝试替代工具。")
        self.assertEqual(update_agent_metrics({}, {"type": "done"})["tool_calls"], [])

    def test_real_parallel_calls_keep_ids_modes_and_individual_time(self):
        barrier = Barrier(2)
        @tool
        def trace_number(value: int) -> int:
            """两个线程实际会合；顺序执行无法通过屏障。"""
            barrier.wait(timeout=3)
            return value * 2
        calls = [{"name": trace_number.name, "args": {"value": n}, "call_id": str(n)} for n in (3, 5)]
        events = [{"type": "tool_call", "iteration": 1, **c} for c in calls]
        events += [{**e, "iteration": 1} for e in execute_calls(calls, [trace_number], parallel=True)]
        result = self.collect(events)
        self.assertEqual([t["result"] for t in result["trace"] if t["type"] == "tool_result"], [6, 10])
        self.assertEqual(result["tools"]["success_rate"], 1)
        self.assertTrue(all(c["execution_mode"] == "parallel" and c["seconds"] > 0 for c in result["tool_calls"]))
        self.assertAlmostEqual(result["tools"]["mean_seconds"], sum(c["seconds"] for c in result["tool_calls"]) / 2)

    def test_real_completed_timeout_retry_keeps_attempts_and_one_success(self):
        invoked = []
        @tool
        def trace_retry() -> str:
            """故障注入：第一次已结束超时，第二次实际成功。"""
            invoked.append(1)
            if len(invoked) == 1:
                raise TimeoutError("一次已结束超时")
            return "已恢复"
        config = deepcopy(load_config())
        config["agent"].update(tool_timeout_seconds=3, max_tool_retries=1)
        with patch("src.agent.router.load_config", return_value=config):
            events = list(execute_calls([{"name": trace_retry.name, "args": {}, "call_id": "retry"}], [trace_retry]))
        result = self.collect(events)
        self.assertEqual(len(invoked), 2)
        self.assertEqual(result["tools"]["completed"], 1)
        self.assertEqual(result["tools"]["success_rate"], 1)
        self.assertEqual(result["tool_calls"][0]["attempts"], 2)
        self.assertEqual([a["status"] for a in result["trace"][0]["attempts"]], ["error", "success"])


class TestAgentMetricsEntryAndPage(unittest.TestCase):
    """实际SQLite、JSONL和Streamlit页面；阶段事件模拟，不启动外部模型。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = deepcopy(load_config())
        self.config["paths"]["logs"] = str(Path(self.directory.name) / "logs")
        self.config["paths"]["session_db"] = str(Path(self.directory.name) / "memory.sqlite3")
        for module in ("src.utils.config", "src.utils.logger", "src.agent.memory"):
            patcher = patch(module + ".load_config", return_value=self.config)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("src.agent.react_loop._run_react", side_effect=lambda *args: iter(deepcopy(TestAgentMetrics.events())))
        self.core = patcher.start()
        self.addCleanup(patcher.stop)

    def logs(self):
        return [json.loads(line) for path in Path(self.config["paths"]["logs"]).glob("agent_*.jsonl")
                for line in path.read_text().splitlines()]

    def test_session_persists_incremental_metrics_identity_and_answer(self):
        memory = MemoryManager()
        session = memory.create_session("alice")
        events = list(run_session("计算", "alice", session, memory=memory))
        records = self.logs()
        self.assertEqual(len(records), len(events))
        self.assertEqual(records[-1]["metrics"]["tokens"]["total"], 161)
        self.assertEqual(records[0]["metrics"]["tokens"]["total"], 12)
        self.assertTrue(all(r["user_id"] == "alice" and r["session_id"] == session for r in records))
        self.assertEqual(memory.get_messages("alice", session)[-1].content, "结果为6。")
        self.assertGreaterEqual(records[-1]["metrics"]["response_seconds"], records[0]["metrics"]["response_seconds"])

    def test_log_failure_does_not_destroy_successful_answer(self):
        memory = MemoryManager()
        with patch("src.utils.logger.record_agent_request", side_effect=OSError("只读目录")):
            events = list(run_session("计算", "alice", memory.create_session("alice"), memory=memory))
        self.assertTrue(events[-1]["task_complete"])
        self.assertIn("只读目录", events[-1]["log_error"])

    def test_failed_summary_is_not_hidden_from_request_usage(self):
        self.config["memory"].update(summary_trigger_turns=4, summary_keep_recent_turns=2, summary_max_tokens=120)
        memory = MemoryManager()
        session = memory.create_session("alice")
        for index in range(4):
            memory.append_turn("alice", session, f"旧问题{index}", f"旧回答{index}")
        with patch("src.agent.memory._summarize", side_effect=RuntimeError("摘要模型失败")):
            events = list(run_session("计算", "alice", session, memory=memory))
        tokens = events[-1]["metrics"]["tokens"]
        self.assertIsNone(tokens["total"])
        self.assertEqual(tokens["known_total"], 161)
        self.assertEqual(tokens["unknown_calls"], 1)
        self.assertEqual(events[0]["metrics"]["calls"][0]["phase"], "记忆摘要")
        self.assertEqual(len(memory.get_messages("alice", session)), 10)

    def test_close_after_first_event_logs_partial_request_without_half_turn(self):
        memory = MemoryManager()
        session = memory.create_session("alice")
        stream = run_session("计算", "alice", session, memory=memory)
        next(stream)
        stream.close()
        self.assertEqual(len(self.logs()), 1)
        self.assertEqual(memory.get_messages("alice", session), [])

    def page(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(Path(__file__).resolve().parents[2] / "src/frontend/app.py"), default_timeout=10).run()

    def test_page_is_lazy_and_shows_tool_tokens_without_reexecuting_on_rerun(self):
        app = self.page()
        self.assertFalse(app.exception)
        self.core.assert_not_called()
        self.assertTrue(Path(self.config["paths"]["session_db"]).exists())
        self.assertEqual(len(app.session_state["agent_memory"].list_sessions(app.session_state["agent_user_id"])), 1)
        app.text_input(key="agent_question").set_value("计算")
        app.button(key="run_agent").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(next(m.value for m in app.metric if m.label == "本次 Agent Token"), "161")
        self.assertTrue(any("calculator + current_time" in str(t.value) for t in app.dataframe))
        count = len(self.logs())
        app.run()
        self.core.assert_called_once()
        self.assertEqual(len(self.logs()), count)
        self.assertEqual(app.session_state["agent_last_event"]["type"], "done")

    def test_page_unknown_usage_and_incomplete_answer_are_both_visible(self):
        events = TestAgentMetrics.events()
        events[3]["result"] = None
        events[-1].update(task_complete=False, stop_reason="incomplete")
        self.core.side_effect = lambda *args: iter(deepcopy(events))
        app = self.page()
        app.text_input(key="agent_question").set_value("计算")
        app.button(key="run_agent").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(next(m.value for m in app.metric if m.label == "本次 Agent Token"), "未知")
        self.assertTrue(any("结果为6。" in m.value for m in app.markdown))
        self.assertTrue(any("任务未完成" in w.value for w in app.warning))

    def test_session_log_trace_is_incremental_and_request_isolated(self):
        memory = MemoryManager()
        session = memory.create_session("alice")
        first = list(run_session("计算", "alice", session, memory=memory))
        records = self.logs()
        self.assertEqual([len(r["metrics"]["trace"]) for r in records], list(range(1, 8)))
        self.assertEqual(records[-1]["metrics"]["trace"], first[-1]["metrics"]["trace"])
        second = list(run_session("追问", "alice", session, memory=memory))
        self.assertEqual(len(second[-1]["metrics"]["trace"]), 7)
        self.assertNotEqual(first[-1]["request_id"], second[-1]["request_id"])

    def test_page_shows_round_trace_stats_and_actual_error_without_reexecution(self):
        events = TestAgentMetrics.events()
        events[3]["elapsed_seconds"] = 0.2
        events[4].update(status="error", error="查询超时", error_kind="deadline", pending=True,
                         elapsed_seconds=0.4, attempts=[{"status": "running"}])
        events[-1].update(task_complete=False, stop_reason="tool_timeout")
        self.core.side_effect = lambda *args: iter(deepcopy(events))
        app = self.page()
        app.text_input(key="agent_question").set_value("计算")
        app.button(key="run_agent").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(next(m.value for m in app.metric if m.label == "本次工具调用成功率"), "50.0%")
        self.assertEqual(next(m.value for m in app.metric if m.label == "平均工具耗时"), "0.300 秒")
        self.assertTrue(any("第1轮" in e.label for e in app.expander))
        for text in ("Thought · 决策说明", "Action · calculator", "工具返回 · current_time · 失败", "Observation · 结果判断"):
            self.assertTrue(any(text in m.value for m in app.markdown), text)
        self.assertTrue(any("后台工具可能仍在运行" in w.value for w in app.warning))
        count = len(self.logs())
        app.run()
        self.assertFalse(app.exception)
        self.core.assert_called_once()
        self.assertEqual(len(self.logs()), count)

    def test_page_direct_answer_has_trace_and_no_tool_rate(self):
        events = [{"type": "thought", "iteration": 1, "thought": "直接回答。", "usage": {"prompt_eval_count": 1, "eval_count": 1}},
                  {"type": "action_skipped", "iteration": 1, "reason": "无需工具。"},
                  {"type": "observation", "iteration": 1, "observation": "可以回答。", "usage": {"prompt_eval_count": 1, "eval_count": 1}},
                  {"type": "done", "iterations": 1, "task_complete": True, "stop_reason": "task_complete", "full_response": "回答。"}]
        self.core.side_effect = lambda *args: iter(deepcopy(events))
        app = self.page()
        app.text_input(key="agent_question").set_value("解释术语")
        app.button(key="run_agent").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(next(m.value for m in app.metric if m.label == "本次工具调用成功率"), "暂无已返回调用")
        self.assertTrue(any("Action · 已跳过" in m.value for m in app.markdown))


    def test_central_chat_and_graph_survive_rerun_without_duplicate(self):
        """中央展示真实问答，底部按调用ID画图；页面重跑不重复生成或追加历史。"""
        app = self.page()
        app.text_input(key="agent_question").set_value("计算")
        app.button(key="run_agent").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_message), 2)
        self.assertEqual(app.session_state["agent_messages"][0]["answer"], "结果为6。")
        self.assertTrue(app.get("graphviz_chart"))
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_message), 2)
        self.assertEqual(len(app.session_state["agent_messages"]), 1)
        self.core.assert_called_once()
        app.text_input(key="agent_question").set_value("再次计算")
        app.button(key="run_agent").click().run()
        self.assertEqual(len(app.chat_message), 4)
        self.assertEqual(len(app.session_state["agent_messages"]), 2)

    def test_new_switch_and_followup_use_only_selected_history(self):
        """新会话无旧轨迹；切回A后续问的Context只包含A原始问答。"""
        app = self.page()
        a = app.session_state["agent_session_id"]
        app.text_input(key="agent_question").set_value("会话A研究代号")
        app.button(key="run_agent").click().run()
        self.assertIn("会话A研究代号", app.selectbox(key="conversation_select").options[0])
        app.button(key="new_conversation").click().run()
        b = app.session_state["agent_session_id"]
        self.assertNotEqual(a, b)
        self.assertEqual(app.session_state["agent_messages"], [])
        self.assertNotIn("agent_last_event", app.session_state)
        self.assertEqual(app.text_input(key="agent_question").value, "")
        app.text_input(key="agent_question").set_value("会话B研究代号")
        app.button(key="run_agent").click().run()
        app.text_input(key="agent_question").set_value("会话B未提交草稿")
        app.selectbox(key="conversation_select").set_value(a).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_message), 2)
        self.assertEqual(app.session_state["agent_messages"][0]["question"], "会话A研究代号")
        self.assertEqual(app.text_input(key="agent_question").value, "")
        self.assertEqual(self.core.call_count, 2)
        app.text_input(key="agent_question").set_value("继续会话A")
        app.button(key="run_agent").click().run()
        context = self.core.call_args.args[2]
        self.assertEqual(context["history"][0]["content"], "会话A研究代号")
        self.assertNotIn("会话B研究代号", json.dumps(context, ensure_ascii=False))

    def test_refresh_keeps_identity_history_incomplete_status_and_trace(self):
        from streamlit.testing.v1 import AppTest
        events = TestAgentMetrics.events()
        events[-1].update(task_complete=False, stop_reason="incomplete")
        self.core.side_effect = lambda *args: iter(deepcopy(events))
        app = self.page()
        app.text_input(key="agent_question").set_value("未完成请求")
        app.button(key="run_agent").click().run()
        refreshed = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "src/frontend/app.py"), default_timeout=10)
        refreshed.query_params.update(app.query_params)
        refreshed.run()
        self.assertFalse(refreshed.exception)
        self.assertEqual(refreshed.session_state["agent_session_id"], app.session_state["agent_session_id"])
        self.assertEqual(refreshed.session_state["agent_user_id"], app.session_state["agent_user_id"])
        self.assertEqual(len(refreshed.chat_message), 2)
        self.assertFalse(refreshed.session_state["agent_messages"][0]["complete"])
        self.assertTrue(any("任务未完成" in w.value for w in refreshed.warning))
        self.assertTrue(refreshed.get("graphviz_chart"))
        self.core.assert_called_once()

    def test_cancel_delete_archive_restore_preserve_history(self):
        app = self.page()
        a = app.session_state["agent_session_id"]
        app.text_input(key="agent_question").set_value("待回收历史")
        app.button(key="run_agent").click().run()
        app.button(key="new_conversation").click().run()
        b = app.session_state["agent_session_id"]
        app.selectbox(key="conversation_select").set_value(a).run()
        app.button(key="delete_conversation").click().run()
        app.button(key="cancel_delete_conversation").click().run()
        self.assertEqual(app.session_state["agent_session_id"], a)
        app.button(key="delete_conversation").click().run()
        app.button(key="confirm_delete_conversation").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["agent_session_id"], b)
        self.assertEqual(len(app.selectbox(key="conversation_select").options), 1)
        self.assertEqual(app.selectbox(key="conversation_select").value, b)
        memory, user = app.session_state["agent_memory"], app.session_state["agent_user_id"]
        self.assertEqual(memory.list_sessions(user, archived=True), [a])
        app.button(key="restore_conversation").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["agent_session_id"], a)
        self.assertEqual(len(app.chat_message), 2)
        self.core.assert_called_once()

    def test_last_deleted_session_is_replaced_and_foreign_url_is_not_loaded(self):
        from streamlit.testing.v1 import AppTest
        app = self.page()
        old = app.session_state["agent_session_id"]
        memory = app.session_state["agent_memory"]
        foreign = memory.create_session("another-user")
        memory.append_turn("another-user", foreign, "其他用户私有问题", "其他用户私有回答")
        refreshed = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "src/frontend/app.py"), default_timeout=10)
        refreshed.query_params.update(app.query_params)
        refreshed.query_params["conversation"] = foreign
        refreshed.run()
        self.assertEqual(refreshed.session_state["agent_session_id"], old)
        self.assertEqual(len(refreshed.selectbox(key="conversation_select").options), 1)
        self.assertEqual(refreshed.selectbox(key="conversation_select").value, old)
        self.assertFalse(refreshed.chat_message)
        app.button(key="delete_conversation").click().run()
        app.button(key="confirm_delete_conversation").click().run()
        self.assertFalse(app.exception)
        self.assertNotEqual(app.session_state["agent_session_id"], old)
        self.assertEqual(app.session_state["agent_messages"], [])
        self.assertEqual(len(app.selectbox(key="conversation_select").options), 1)
        self.core.assert_not_called()

    def test_delete_database_failure_keeps_current_session_and_retry_works(self):
        app = self.page()
        old = app.session_state["agent_session_id"]
        app.button(key="delete_conversation").click().run()
        with patch("src.agent.memory.MemoryManager.delete_session", side_effect=sqlite3.OperationalError("只读数据库")):
            app.button(key="confirm_delete_conversation").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["agent_session_id"], old)
        self.assertTrue(any("只读数据库" in e.value for e in app.error))
        self.assertTrue(app.button(key="run_agent").disabled)
        app.button(key="confirm_delete_conversation").click().run()
        self.assertFalse(app.exception)
        self.assertNotEqual(app.session_state["agent_session_id"], old)


if __name__ == "__main__":
    unittest.main()
