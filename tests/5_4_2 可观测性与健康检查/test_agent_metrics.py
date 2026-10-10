"""5.4.2 可观测性与健康检查：TestAgentMetrics、TestAgentTraceMetrics、TestAgentMetricsEntryAndPage。"""

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
from langchain_core.messages import AIMessage
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


class TestAgentMetricsEntryAndPage(unittest.TestCase):
    """实际SQLite、JSONL和Streamlit页面；阶段事件模拟，不启动外部模型。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = deepcopy(load_config())
        self.config["paths"]["logs"] = str(Path(self.directory.name) / "logs")
        self.config["paths"]["session_db"] = str(Path(self.directory.name) / "memory.sqlite3")
        health_patcher = patch("src.utils.config.check_health", return_value={
            "llm": {"status": "ok"}, "vector_database": {"status": "ok"}})
        health_patcher.start()
        self.addCleanup(health_patcher.stop)
        for module in ("src.utils.config", "src.utils.logger", "src.agent.memory"):
            patcher = patch(module + ".load_config", return_value=self.config)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("src.agent.react_loop._run_react", side_effect=lambda *args, **kwargs: iter(deepcopy(TestAgentMetrics.events())))
        self.core = patcher.start()
        self.addCleanup(patcher.stop)

    def logs(self):
        return [json.loads(line) for path in Path(self.config["paths"]["logs"]).glob("agent_*.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines()]


    def page(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(Path(__file__).resolve().parents[2] / "src/frontend/app.py"), default_timeout=10).run()


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
        self.assertEqual(sum((button.key or "").startswith("conversation:") for button in refreshed.sidebar.button), 1)
        self.assertEqual(refreshed.button(key=f"conversation:{old}").proto.type, "primary")
        self.assertFalse(refreshed.chat_message)
        app.button(key="delete_conversation").click().run()
        app.button(key="confirm_delete_conversation").click().run()
        self.assertFalse(app.exception)
        self.assertNotEqual(app.session_state["agent_session_id"], old)
        self.assertEqual(app.session_state["agent_messages"], [])
        self.assertEqual(sum((button.key or "").startswith("conversation:") for button in app.sidebar.button), 1)
        self.core.assert_not_called()


if __name__ == "__main__":
    unittest.main()
