"""5.3.3 Agent决策优化：TestRoutingAndParallel。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from copy import deepcopy
from io import BytesIO
import json
from threading import Barrier, Lock
import unittest
from unittest.mock import patch
from langchain_core.tools import tool
from src.agent.react_loop import act, run_react
from src.agent.tools import current_time
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


class TestRoutingAndParallel(unittest.TestCase):
    """路由不调用模型；并发以真实线程与同步屏障证明，HTTP仅隔离模型协议。"""

    def setUp(self):
        self.config = deepcopy(load_config())
        self.addCleanup(patch.stopall)
        for module in ("src.agent.router", "src.agent.react_loop"):
            patch(module + ".load_config", return_value=self.config).start()
        self.invocations = []

        @tool
        def read_number(value: int) -> int:
            """读取已给定整数并返回两倍；用于可核验的独立工具调用。"""
            self.invocations.append(value)
            if value < 0:
                raise ValueError("输入不能为负数")
            return value * 2

        self.tools = [read_number]
        self.plan = {"thought": "两个给定输入互不依赖，可同时读取。", "next_step": "tool", "tool_name": "read_number", "parallel_tools": ["read_number"]}
        self.calls = [{"call_id": "call-a", "name": "read_number", "args": {"value": 3}},
                      {"call_id": "call-b", "name": "read_number", "args": {"value": 5}}]
        self.response = {"model": "qwen2.5:7b", "done": True, "done_reason": "stop", "prompt_eval_count": 300, "eval_count": 60,
                         "message": {"tool_calls": [{"function": {"name": call["name"], "arguments": call["args"]}} for call in self.calls]}}
        self.finished = {"observation": "两个结果均已返回。", "decision": "finish", "task_complete": True, "answer": "结果为6和10。"}

    def packet(self, content):
        return {**self.response, "message": {"content": json.dumps(content)}}

    def action(self, response=None, plan=None, tools=None):
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(response or self.response).encode())) as http:
            events = list(act("独立读取3和5。", plan or self.plan, tools or self.tools))
        return events, http


    def test_rule_path_skips_thought_model_but_keeps_action_and_observation(self):
        response = {**self.response, "message": {"tool_calls": [{"function": {"name": "current_time", "arguments": {}}}]}}
        with patch("src.agent.react_loop.think") as think_mock, patch("src.agent.react_loop.urlopen", side_effect=[
                BytesIO(json.dumps(response).encode()), BytesIO(json.dumps(self.packet(self.finished)).encode())]) as http:
            events = list(run_react("返回当前时间", [current_time]))
        think_mock.assert_not_called()
        self.assertEqual(http.call_count, 2)
        self.assertEqual(events[0]["route"], "rule")
        self.assertEqual(events[1]["name"], "current_time")
        self.assertTrue(events[-1]["task_complete"])


    def test_two_threads_reach_barrier_and_return_stable_order(self):
        barrier, lock, active, maximum = Barrier(2), Lock(), [0], [0]
        @tool("read_number")
        def read_number(value: int) -> int:
            """屏障要求两个真实调用同时到达，顺序执行会超时并失败。"""
            with lock:
                active[0] += 1
                maximum[0] = max(maximum[0], active[0])
            try:
                barrier.wait(timeout=3)
                return value * 2
            finally:
                with lock:
                    active[0] -= 1
        results = list(execute_calls(self.calls, [read_number], parallel=True))
        self.assertEqual(maximum[0], 2)
        self.assertEqual([item["result"] for item in results], [6, 10])
        self.assertEqual([item["call_id"] for item in results], ["call-a", "call-b"])
        self.assertTrue(all(item["execution_mode"] == "parallel" and item["status"] == "success" for item in results))


    def test_individual_failure_preserves_other_independent_result(self):
        calls = deepcopy(self.calls)
        calls[0]["args"]["value"] = -1
        results = list(execute_calls(calls, self.tools, parallel=True))
        self.assertEqual([item["status"] for item in results], ["error", "success"])
        self.assertIn("输入不能为负数", results[0]["error"])
        self.assertEqual(results[1]["result"], 10)
        self.assertCountEqual(self.invocations, [-1, 5])


if __name__ == "__main__":
    unittest.main()
