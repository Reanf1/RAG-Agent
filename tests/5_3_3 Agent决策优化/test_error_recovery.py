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


if __name__ == "__main__":
    unittest.main()
