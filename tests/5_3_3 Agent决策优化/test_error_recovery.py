"""5.3.3 Agent决策优化：TestErrorRecovery。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from copy import deepcopy
from io import BytesIO
import json
from threading import Event
import unittest
from unittest.mock import patch
from urllib.error import URLError
from langchain_core.tools import tool
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


class TestErrorRecovery(unittest.TestCase):
    """故障显式注入，工具和线程实际执行；只隔离模型HTTP，不冒充论文质量评测。"""

    def setUp(self):
        self.config = deepcopy(load_config())
        self.config["agent"].update(tool_timeout_seconds=1, max_tool_retries=1,
                                    max_repeated_calls=2, max_iterations=6)
        for target in ("src.agent.router.load_config", "src.agent.react_loop.load_config"):
            patcher = patch(target, return_value=self.config)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.invocations, self.mode = [], "ok"
        @tool
        def primary_lookup(key: str) -> dict:
            """查询给定词条；测试注入一次超时、持续超时、执行异常或输入错误。"""
            self.invocations.append(("primary_lookup", key))
            if self.mode == "timeout" or self.mode == "once" and len(self.invocations) == 1:
                raise TimeoutError("明确注入的已结束超时")
            if self.mode == "execution":
                raise RuntimeError("明确注入的主查询故障")
            if self.mode == "input":
                raise ValueError("请提供正确词条")
            return {"key": key}
        @tool
        def backup_lookup(key: str) -> dict:
            """用备用查询获取同一给定词条，用于验证替代真实执行和参数关联。"""
            self.invocations.append(("backup_lookup", key))
            return {"key": key}
        self.tools = [primary_lookup, backup_lookup]
        self.call = {"name": "primary_lookup", "args": {"key": "Transformer"}, "call_id": "primary-id"}
        self.pending = {"observation": "还需获取词条。", "decision": "continue", "task_complete": False, "answer": ""}
        self.failed = {"observation": "查询失败。", "decision": "finish", "task_complete": False, "answer": "查询失败，请重试。"}
        self.finished = {"observation": "词条已获取。", "decision": "finish", "task_complete": True, "answer": "词条为Transformer。"}

    def packet(self, content=None, name=None, args=None):
        message = {"content": json.dumps(content, ensure_ascii=False)}
        if name:
            message = {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args or self.call["args"]}}]}
        return {"model": "qwen2.5:7b", "message": message, "done": True,
                "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 20}

    def plan(self, name=None):
        return self.packet(name=name) if name else self.packet("说明结果或失败。")

    def loop(self, packets, tools=None, context=None):
        with patch("src.agent.react_loop.route_question", return_value=None), \
                patch("src.agent.react_loop.urlopen", side_effect=[BytesIO(json.dumps(item).encode()) for item in packets]) as http:
            events = list(run_react("查询Transformer词条。", self.tools if tools is None else tools, context))
        return events, http

    def test_completed_timeout_retries_once_and_preserves_both_attempts(self):
        self.mode = "once"
        result = list(execute_calls([self.call], self.tools))[0]
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(self.invocations), 2)
        self.assertEqual([item["status"] for item in result["attempts"]], ["error", "success"])
        self.assertEqual(result["attempts"][0]["error_kind"], "timeout")
        self.assertIn("已结束超时", result["attempts"][0]["error"])
        self.assertEqual(result["message"].tool_call_id, "primary-id")
        self.assertEqual(result["message"].status, "success")

    def test_exhausted_timeout_returns_real_error_without_third_attempt(self):
        self.mode = "timeout"
        result = list(execute_calls([self.call], self.tools))[0]
        self.assertEqual(result["error_kind"], "timeout")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(len(self.invocations), 2)
        self.assertEqual(result["message"].status, "error")

    def test_retry_can_be_disabled(self):
        self.mode = "timeout"
        self.config["agent"]["max_tool_retries"] = 0
        self.assertEqual(len(list(execute_calls([self.call], self.tools))[0]["attempts"]), 1)
        self.assertEqual(len(self.invocations), 1)

    def test_non_timeout_input_or_execution_error_is_not_retried(self):
        for mode, kind in (("input", "input"), ("execution", "execution")):
            self.mode, self.invocations = mode, []
            result = list(execute_calls([self.call], self.tools))[0]
            self.assertEqual(result["error_kind"], kind)
            self.assertEqual(len(self.invocations), 1)

    def test_wrapped_timeout_is_classified_without_guessing_from_text(self):
        @tool
        def wrapped_timeout() -> None:
            """模拟已有模型请求将真实超时包装为RuntimeError。"""
            try:
                raise URLError(TimeoutError("底层超时"))
            except URLError as error:
                raise RuntimeError("请求失败") from error
        result = list(execute_calls([{"name": "wrapped_timeout", "args": {}, "call_id": "wrapped"}], [wrapped_timeout]))[0]
        self.assertEqual(result["error_kind"], "timeout")
        self.assertEqual(len(result["attempts"]), 2)
        @tool
        def timeout_text() -> None:
            """字符串中包含timeout，不代表真实超时。"""
            raise RuntimeError("timeout只是日志文字")
        result = list(execute_calls([{"name": "timeout_text", "args": {}, "call_id": "text"}], [timeout_text]))[0]
        self.assertEqual(result["error_kind"], "execution")
        self.assertEqual(len(result["attempts"]), 1)

    def blocked_tool(self, timeout_after_release=False):
        started, released, finished = Event(), Event(), Event()
        self.addCleanup(released.set)
        @tool
        def blocked_lookup(key: str) -> str:
            """真实等待释放事件；让外部等待预算先到期，不把sleep计时误当作取消成功。"""
            self.invocations.append(("blocked_lookup", key))
            started.set()
            try:
                released.wait(timeout=3)
                if timeout_after_release:
                    raise TimeoutError("超期之后才返回的错误")
                return "迟到结果"
            finally:
                finished.set()
        return blocked_lookup, started, released, finished

    def test_deadline_workers_remain_bounded_across_requests(self):
        """连续请求的迟到线程仍占共享执行额度，不能每次超时再开一条线程。"""
        self.config["agent"].update(max_parallel_calls=1, tool_timeout_seconds=.03)
        blocking, _, released, finished = self.blocked_tool()
        try:
            outcomes = [list(execute_calls([{**self.call, "name": blocking.name, "call_id": str(i)}], [blocking]))[0]
                        for i in range(3)]
            self.assertEqual(len(self.invocations), 1)
            self.assertEqual([item["error_kind"] for item in outcomes], ["deadline", "capacity", "capacity"])
            self.assertTrue(all(not item["pending"] for item in outcomes[1:]))
        finally:
            released.set()
            self.assertTrue(finished.wait(timeout=3))
        self.assertEqual(list(execute_calls([self.call], self.tools))[0]["status"], "success")

    def test_deadline_returns_before_worker_exits_and_ignores_late_result(self):
        self.config["agent"]["tool_timeout_seconds"] = .05
        blocking, started, released, finished = self.blocked_tool()
        call = {**self.call, "name": blocking.name}
        try:
            result = list(execute_calls([call], [blocking]))[0]
            self.assertTrue(started.is_set())
            self.assertFalse(finished.is_set())
            self.assertTrue(result["pending"])
            self.assertEqual(result["error_kind"], "deadline")
            self.assertIsNone(result["result"])
            snapshot = deepcopy(result)
        finally:
            released.set()
            self.assertTrue(finished.wait(timeout=3))
        self.assertEqual(result, snapshot)
        self.assertEqual(len(self.invocations), 1)

    def test_expired_request_never_starts_delayed_first_attempt(self):
        """线程已提交但未进入工具时请求到期，之后恢复调度不能再启动首次调用。"""
        from concurrent.futures import ThreadPoolExecutor
        ready, released, finished = Event(), Event(), Event()
        self.addCleanup(released.set)
        self.config["agent"]["tool_timeout_seconds"] = .03

        class DelayedExecutor(ThreadPoolExecutor):
            def submit(executor, function, *args, **kwargs):
                def delayed():
                    ready.set()
                    try:
                        released.wait(timeout=3)
                        return function(*args, **kwargs)
                    finally:
                        finished.set()
                return super().submit(delayed)

        with patch("src.agent.router.ThreadPoolExecutor", DelayedExecutor):
            try:
                result = list(execute_calls([self.call], self.tools))[0]
                self.assertTrue(ready.is_set())
                self.assertEqual(result["error_kind"], "deadline")
                self.assertFalse(self.invocations)
            finally:
                released.set()
                self.assertTrue(finished.wait(timeout=3))
        self.assertFalse(self.invocations)

    def test_late_timeout_does_not_start_background_retry(self):
        self.config["agent"]["tool_timeout_seconds"] = .05
        blocking, _, released, finished = self.blocked_tool(timeout_after_release=True)
        try:
            results = list(execute_calls([{**self.call, "name": blocking.name}], [blocking]))
            self.assertTrue(results[0]["pending"])
        finally:
            released.set()
            self.assertTrue(finished.wait(timeout=3))
        self.assertEqual(len(self.invocations), 1)

    def test_parallel_timeout_preserves_success_and_shared_deadline(self):
        self.config["agent"]["tool_timeout_seconds"] = .05
        blocking, _, released, finished = self.blocked_tool()
        calls = [{**self.call, "name": blocking.name}, {**self.call, "name": "backup_lookup", "call_id": "backup-id"}]
        try:
            results = list(execute_calls(calls, [blocking, self.tools[1]], parallel=True))
            self.assertEqual([r["status"] for r in results], ["error", "success"])
            self.assertEqual(results[1]["result"], {"key": "Transformer"})
            self.assertEqual(results[1]["message"].tool_call_id, "backup-id")
        finally:
            released.set()
            self.assertTrue(finished.wait(timeout=3))

    def test_closing_batch_does_not_wait_or_allow_late_retry(self):
        blocking, started, released, finished = self.blocked_tool(timeout_after_release=True)
        calls = [self.call, {**self.call, "name": blocking.name, "call_id": "blocked-id"}]
        stream = execute_calls(calls, [self.tools[0], blocking], parallel=True)
        try:
            self.assertEqual(next(stream)["status"], "success")
            self.assertTrue(started.wait(timeout=1))
            stream.close()
            self.assertFalse(finished.is_set())
        finally:
            released.set()
            self.assertTrue(finished.wait(timeout=3))
            stream.close()
        self.assertEqual(len(self.invocations), 2)

    def test_agent_deadline_stops_without_observation_model_or_next_round(self):
        self.config["agent"]["tool_timeout_seconds"] = .05
        blocking, _, released, finished = self.blocked_tool()
        try:
            events, http = self.loop([self.packet(name=blocking.name)], [blocking, self.tools[1]])
            self.assertEqual(http.call_count, 1)
            self.assertEqual(events[-1]["stop_reason"], "tool_timeout")
            self.assertFalse(events[-1]["task_complete"])
            self.assertIn("后台函数", events[-1]["full_response"])
            self.assertTrue(events[-1]["context"]["observations"][0]["pending"])
        finally:
            released.set()
            self.assertTrue(finished.wait(timeout=3))

    def test_execution_error_forces_alternative_planning_and_preserves_failed_result(self):
        self.mode = "execution"
        context = {"observations": []}
        original = deepcopy(context)
        events, http = self.loop([self.packet(name="primary_lookup"), self.packet(self.failed),
                                  self.packet(name="backup_lookup"), self.packet(self.finished)], context=context)
        self.assertTrue(events[-1]["task_complete"])
        self.assertEqual([item[0] for item in self.invocations], ["primary_lookup", "backup_lookup"])
        self.assertEqual([item["status"] for item in events[-1]["context"]["observations"]], ["error", "success"])
        self.assertEqual(len([e for e in events if e["type"] == "recovery"]), 1)
        specs = json.loads(json.loads(http.call_args_list[2].args[0].data)["messages"][0]["content"].split("【可用工具描述】\n")[1].splitlines()[0])
        self.assertEqual([item["name"] for item in specs["available_tools"]], ["backup_lookup"])
        results = [e for e in events if e["type"] == "tool_result"]
        self.assertNotEqual(results[0]["call_id"], results[1]["call_id"])
        self.assertEqual(context, original)

    def test_exhausted_timeout_can_use_alternative(self):
        self.mode = "timeout"
        events, _ = self.loop([self.packet(name="primary_lookup"), self.packet(self.failed),
                              self.packet(name="backup_lookup"), self.packet(self.finished)])
        self.assertTrue(events[-1]["task_complete"])
        self.assertEqual(len(self.invocations), 3)
        self.assertEqual(len(events[-1]["context"]["observations"][0]["attempts"]), 2)

    def test_no_alternative_reports_failure_and_does_not_invent_tools(self):
        self.mode = "execution"
        events, http = self.loop([self.packet(name="primary_lookup"), self.packet(self.failed)], self.tools[:1])
        self.assertFalse(events[-1]["task_complete"])
        self.assertEqual(events[-1]["stop_reason"], "incomplete")
        self.assertEqual(http.call_count, 2)
        self.assertFalse(any(e["type"] == "recovery" for e in events))

    def test_all_alternatives_fail_and_stop_without_reusing_failed_tools(self):
        self.mode = "execution"
        @tool("backup_lookup")
        def unavailable_backup(key: str) -> dict:
            """备用也真实抛错，验证恢复次数受注册工具和迭代上限约束。"""
            self.invocations.append(("backup_lookup", key))
            raise RuntimeError("备用查询也不可用")
        events, _ = self.loop([self.packet(name="primary_lookup"), self.packet(self.failed),
                              self.packet(name="backup_lookup"), self.packet(self.failed)],
                             [self.tools[0], unavailable_backup])
        self.assertEqual(events[-1]["stop_reason"], "incomplete")
        self.assertEqual(len(self.invocations), 2)
        self.assertEqual(events[-1]["context"]["recovery"]["available_alternatives"], [])
        self.assertEqual([r["status"] for r in events[-1]["context"]["observations"]], ["error", "error"])

    def test_recovery_cannot_exceed_last_iteration(self):
        self.mode = "execution"
        self.config["agent"]["max_iterations"] = 1
        events, http = self.loop([self.packet(name="primary_lookup"), self.packet(self.pending)])
        self.assertEqual(events[-1]["stop_reason"], "max_iterations")
        self.assertEqual(http.call_count, 2)
        self.assertEqual(len(self.invocations), 1)

    def test_input_error_does_not_force_unrelated_alternative(self):
        self.mode = "input"
        events, http = self.loop([self.packet(name="primary_lookup"), self.packet(self.failed)])
        self.assertEqual(events[-1]["stop_reason"], "incomplete")
        self.assertEqual(http.call_count, 2)
        self.assertEqual(len(self.invocations), 1)

    def test_failed_tool_cannot_be_reselected_in_recovery_round(self):
        self.mode = "execution"
        events, http = self.loop([self.packet(name="primary_lookup"), self.packet(self.failed),
                                  self.plan("primary_lookup")])
        self.assertEqual(events[-1]["stop_reason"], "error")
        self.assertEqual(http.call_count, 3)
        self.assertEqual(len(self.invocations), 1)

    def test_no_suitable_alternative_finishes_incomplete_without_a_tool(self):
        self.mode = "execution"
        events, _ = self.loop([self.packet(name="primary_lookup"), self.packet(self.failed),
                              self.plan(), self.packet(self.failed)])
        self.assertEqual(events[-1]["stop_reason"], "incomplete")
        self.assertEqual(len(self.invocations), 1)
        self.assertTrue(events[-1]["context"]["recovery"]["pending"])

    def test_unrecovered_answer_cannot_claim_completion(self):
        self.mode = "execution"
        events, _ = self.loop([self.packet(name="primary_lookup"), self.packet(self.failed),
                              self.plan(), self.packet(self.finished)])
        self.assertFalse(events[-1]["task_complete"])
        self.assertEqual(events[-1]["stop_reason"], "error")

    def test_same_call_is_stopped_before_third_execution(self):
        round_packets = [self.packet(name="primary_lookup"), self.packet(self.pending)]
        events, http = self.loop([*round_packets, *round_packets, self.packet(name="primary_lookup")])
        self.assertEqual(events[-1]["stop_reason"], "repeated_calls")
        self.assertIn("死循环", events[-1]["full_response"])
        self.assertEqual(len(self.invocations), 2)
        self.assertEqual(http.call_count, 5)

    def test_alternating_cycle_is_also_stopped(self):
        packets = []
        for name in ("primary_lookup", "backup_lookup", "primary_lookup", "backup_lookup"):
            packets.extend([self.packet(name=name), self.packet(self.pending)])
        events, _ = self.loop([*packets, self.packet(name="primary_lookup")])
        self.assertEqual(events[-1]["stop_reason"], "repeated_calls")
        self.assertEqual(len(self.invocations), 4)
        self.assertEqual(events[-1]["iterations"], 5)

    def test_different_arguments_and_separate_requests_are_not_a_cycle(self):
        packets = []
        for key in ("A", "B", "C"):
            packets.extend([self.packet(name="primary_lookup", args={"key": key}),
                            self.packet(self.finished if key == "C" else self.pending)])
        self.assertTrue(self.loop(packets)[0][-1]["task_complete"])
        self.invocations = []
        packets = [self.packet(name="primary_lookup"), self.packet(self.finished)]
        for _ in range(3):
            self.assertTrue(self.loop(packets)[0][-1]["task_complete"])
        self.assertEqual(len(self.invocations), 3)

    def test_invalid_recovery_limits_stop_before_model_or_tools(self):
        cases = {"tool_timeout_seconds": (0, -1, True, "1", float("nan"), float("inf")),
                 "max_tool_retries": (-1, True, 1.5, "1"), "max_repeated_calls": (0, -1, True, 1.5)}
        original = deepcopy(self.config["agent"])
        for key, values in cases.items():
            for value in values:
                self.config["agent"] = {**original, key: value}
                with self.subTest(key=key, value=value):
                    events, http = self.loop([])
                    self.assertEqual(events[-1]["iterations"], 0)
                    self.assertFalse(events[-1]["task_complete"])
                    http.assert_not_called()
        self.assertEqual(self.invocations, [])


if __name__ == "__main__":
    unittest.main()
