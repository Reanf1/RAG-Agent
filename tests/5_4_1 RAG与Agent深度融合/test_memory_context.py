"""5.4.1 RAG与Agent深度融合：TestAgentMemoryContext。"""

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


class TestAgentMemoryContext(unittest.TestCase):
    """真实SQLite/消息适配及跨轮工具执行；模型HTTP仅用于确定性功能验证。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = deepcopy(load_config())
        self.config["memory"]["summary_trigger_turns"] = 100
        for module in ("src.agent.memory", "src.agent.react_loop", "src.agent.router", "src.agent.tools"):
            patcher = patch(module + ".load_config", return_value=self.config)
            patcher.start()
            self.addCleanup(patcher.stop)
        from src.agent import MemoryManager as PublicMemory
        self.path = Path(self.directory.name) / "memory.sqlite3"
        self.memory = PublicMemory(self.path)
        self.session = self.memory.create_session("alice")

    def packet(self, content=None, calls=None):
        return BytesIO(json.dumps({"model": "qwen2.5:7b", "done": True, "done_reason": "stop",
            "prompt_eval_count": 100, "eval_count": 20, "message": {"tool_calls": calls} if calls is not None
            else {"content": json.dumps(content, ensure_ascii=False)}}).encode())


    def test_two_round_public_entry_uses_memory_for_calculator_action(self):
        from src.agent import run_session as public_run
        answer_plan = {"thought": "记录本轮实验系数。", "next_step": "answer", "tool_name": None, "parallel_tools": []}
        first = {"observation": "已记录系数。", "decision": "finish", "task_complete": True, "answer": "实验系数为3.14。"}
        with patch("src.agent.react_loop.urlopen", side_effect=[self.packet(answer_plan), self.packet(first)]):
            list(public_run("我的实验系数为3.14，请记住。", "alice", self.session, memory=self.memory))
        plan = {"thought": "按历史系数计算。", "next_step": "tool", "tool_name": "calculator", "parallel_tools": []}
        calls = [{"function": {"name": "calculator", "arguments": {"expression": "3.14*2"}}}]
        finish = {"observation": "工具返回6.28。", "decision": "finish", "task_complete": True, "answer": "结果为6.28。"}
        with patch("src.agent.react_loop.urlopen", side_effect=[self.packet(calls=calls), self.packet(finish)]) as model:
            events = list(public_run("用刚才的系数乘以2。", "alice", self.session, memory=MemoryManager(self.path)))
        self.assertEqual(next(e["result"]["result"] for e in events if e["type"] == "tool_result"), "6.28")
        for call in model.call_args_list:
            self.assertIn("实验系数为3.14", call.args[0].data.decode())
        archived = self.memory.get_messages("alice", self.session)
        self.assertEqual([m.type for m in archived], ["human", "ai"] * 2)
        self.assertNotIn("tool_result", " ".join(m.content for m in archived))
        self.assertTrue(all(e["session_id"] == self.session and e["user_id"] == "alice" for e in events))


if __name__ == "__main__":
    unittest.main()
