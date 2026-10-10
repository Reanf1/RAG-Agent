"""5.3.1 Agent核心循环：TestThought。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from copy import deepcopy
from io import BytesIO
import json
import unittest
from unittest.mock import patch
from langchain_core.tools import tool
from src.agent.react_loop import think
from src.utils.config import load_config
from tests.helpers import isolated_agent_logs


def setUpModule():
    """保持既有 Agent 测试的模块级日志隔离。"""
    global _log_context
    _log_context = isolated_agent_logs()
    _log_context.__enter__()


def tearDownModule():
    _log_context.__exit__(None, None, None)


class TestThought(unittest.TestCase):
    """验证问题/状态传入、计划校验和只规划不执行的边界。"""

    def setUp(self):
        self.config = deepcopy(load_config())
        self.invocations = []

        @tool
        def multiply(a: float, b: float) -> float:
            """计算两数乘积；只用于测试工具描述与不执行的约束。"""
            self.invocations.append((a, b))
            return a * b

        self.tools = [multiply]
        self.plan = {"thought": "下一步用乘法工具获得准确结果。", "next_step": "tool", "tool_name": "multiply"}
        self.response = {"model": "qwen2.5:7b", "message": {"content": "", "tool_calls": [{"function": {"name": "multiply", "arguments": {"a": 3.14, "b": 2.56}}}]},
                         "done": True, "done_reason": "stop", "prompt_eval_count": 400, "eval_count": 30}

    def call(self, response=None, tools=None):
        raw = self.response if response is None else response
        with patch("src.agent.react_loop.load_config", return_value=self.config), \
                patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(raw).encode())):
            return think("3.14乘以2.56", self.tools if tools is None else tools)


    def test_plan_only_selects_registered_tool_and_uses_llm_config(self):
        context = {"observations": [{"tool": "multiply", "result": "已有观察"}]}
        snapshot = deepcopy(context)
        with patch("src.agent.react_loop.load_config", return_value=self.config), \
                patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.response).encode())) as http:
            result = think("3.14乘以2.56", self.tools, context)
        request = http.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, self.config["llm"]["base_url"].rstrip("/") + "/api/chat")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["model"], self.config["llm"]["model"])
        self.assertEqual(payload["options"]["temperature"], self.config["llm"]["temperature"])
        self.assertEqual(payload["tools"][0]["function"]["name"], "multiply")
        params = payload["tools"][0]["function"]["parameters"]
        self.assertEqual(set(params["required"]), {"a", "b"})
        self.assertEqual(params["properties"]["a"]["type"], "number")
        self.assertEqual(params["properties"]["b"]["type"], "number")
        self.assertNotIn("format", payload)
        self.assertEqual(json.loads(payload["messages"][1]["content"])["context"], context)
        self.assertEqual(result["type"], "thought")
        self.assertEqual(result["next_step"], "tool")
        self.assertEqual(result["usage"], {"prompt_eval_count": 400, "eval_count": 30})
        self.assertGreaterEqual(result["elapsed_seconds"], 0)
        self.assertEqual(context, snapshot)
        self.assertEqual(self.invocations, [])  # 选择不是执行；本阶段不能出现Action副作用。
        self.assertEqual(result["tool_calls"], self.response["message"]["tool_calls"])

    def test_direct_answer_without_tools_has_no_tool_calls(self):
        response = {**self.response, "message": {"content": "通用概念可直接回答。"}}
        result = self.call(response, tools=[])
        self.assertEqual(result["next_step"], "answer")
        self.assertIsNone(result["tool_name"])
        self.assertEqual(result["tool_calls"], [])


if __name__ == "__main__":
    unittest.main()
