"""5.3.1 Agent核心循环：TestObservationAndLoop。"""

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
from src.agent.react_loop import observe, run_react
from src.agent.tools import knowledge_base_search
from src.utils.config import load_config
from tests.helpers import isolated_agent_logs


def setUpModule():
    """保持既有 Agent 测试的模块级日志隔离。"""
    global _log_context
    _log_context = isolated_agent_logs()
    _log_context.__enter__()


def tearDownModule():
    _log_context.__exit__(None, None, None)


class TestObservationAndLoop(unittest.TestCase):
    """验证真实结果反馈、多轮任务，以及成功/未完成/错误/上限的区别。"""

    def setUp(self):
        self.config = deepcopy(load_config())
        self.config["agent"]["max_iterations"] = 2
        self.invocations = []

        @tool
        def multiply(a: float, b: float) -> float:
            """计算乘法，用于验证跨轮结果传递。"""
            self.invocations.append(("multiply", a, b))
            return a * b

        @tool
        def add(a: float, b: float) -> float:
            """计算加法，用于完成第二步。"""
            self.invocations.append(("add", a, b))
            return a + b

        @tool
        def divide(a: float, b: float) -> float:
            """实际执行除法，测试除零失败。"""
            self.invocations.append(("divide", a, b))
            return a / b

        self.tools = [multiply, add, divide]
        self.pending = {"observation": "已得到乘积12，还需加5。", "decision": "continue",
                        "task_complete": False, "answer": ""}
        self.finished = {"observation": "已完成两步计算。", "decision": "finish",
                         "task_complete": True, "answer": "结果为17。"}

    def packet(self, content=None, name=None, args=None):
        """只隔离HTTP，保持模型原生消息格式。"""
        message = {"content": json.dumps(content, ensure_ascii=False)}
        if name is not None:
            message = {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}
        return {"model": "qwen2.5:7b", "done": True, "done_reason": "stop", "message": message,
                "prompt_eval_count": 100, "eval_count": 20}

    def plan(self, name=None):
        return self.packet({"thought": "执行下一步。" if name else "已有资料可回答。",
                            "next_step": "tool" if name else "answer", "tool_name": name})

    def http_responses(self, responses):
        return [BytesIO(json.dumps(item, ensure_ascii=False).encode()) if isinstance(item, dict)
                else item for item in responses]

    def run_loop(self, responses, context=None):
        with patch("src.agent.react_loop.load_config", return_value=self.config), \
                patch("src.agent.react_loop.urlopen", side_effect=self.http_responses(responses)) as http:
            events = list(run_react("先计算3乘4，再把乘积加5。", self.tools, context))
        return events, http

    def test_final_answer_uses_one_model_request(self):
        """模型已给出有效最终答案时，观察阶段不再请求改写。"""
        for complete in (True, False):
            with self.subTest(complete=complete):
                packet = self.packet({"answer": "已有资料说明如下。", "task_complete": complete})
                with patch("src.agent.react_loop.urlopen", side_effect=self.http_responses([packet])) as http:
                    events = list(run_react("请解释已有资料。", []))
                self.assertEqual(http.call_count, 1)
                result = events[-1]
                self.assertEqual(result["full_response"], "已有资料说明如下。")
                self.assertEqual(result["task_complete"], complete)
                self.assertEqual(result["metrics"]["tokens"]["input"], 100)
                self.assertEqual(result["metrics"]["tokens"]["output"], 20)


    def test_native_final_cannot_hide_tool_failure(self):
        """直接回答仍需如实保留实际工具失败与未完成状态。"""
        context = {"observations": [{"name": "multiply", "status": "error", "result": None}]}
        thought = {"next_step": "answer", "final": {"answer": "已完成。", "task_complete": True}}
        with patch("src.agent.react_loop.urlopen") as http:
            result = observe("求乘积。", self.tools, context, thought=thought)
        http.assert_not_called()
        self.assertFalse(result["task_complete"])
        self.assertIn("尚未完成", result["answer"])


    def test_two_real_tools_feed_next_round_and_native_observation_messages(self):
        context = {"source": "原始资料", "observations": []}
        original = deepcopy(context)
        events, http = self.run_loop([
            self.packet(name="multiply", args={"a": 3, "b": 4}), self.packet(self.pending),
            self.packet(name="add", args={"a": 12, "b": 5}), self.packet(self.finished)], context)
        self.assertEqual(self.invocations, [("multiply", 3, 4), ("add", 12, 5)])
        self.assertEqual([e["type"] for e in events],
                         ["thought", "tool_call", "tool_result", "observation"] * 2 + ["done"])
        done = events[-1]
        self.assertTrue(done["task_complete"])
        self.assertEqual(done["stop_reason"], "task_complete")
        self.assertEqual(done["iterations"], 2)  # 恰好到达上限时完成，也算成功。
        self.assertEqual(done["full_response"], "结果为17。")
        self.assertEqual([e["result"] for e in done["context"]["observations"]], [12, 17])
        self.assertEqual(done["context"]["source"], "原始资料")
        self.assertEqual(context, original)
        native = json.loads(http.call_args_list[1].args[0].data)["messages"]
        self.assertEqual([m["role"] for m in native], ["system", "user", "assistant", "tool", "user"])
        self.assertEqual(native[2]["tool_calls"][0]["function"], {"name": "multiply", "arguments": {"a": 3, "b": 4}})
        self.assertEqual(native[3]["tool_name"], "multiply")
        self.assertEqual(native[3]["content"], "12.0")
        second_state = json.loads(json.loads(http.call_args_list[2].args[0].data)["messages"][1]["content"])["context"]
        self.assertEqual(second_state["observations"][0]["result"], 12)
        self.assertEqual(second_state["last_observation"]["decision"], "continue")
        self.assertEqual(events[1]["message"].tool_calls[0]["id"], events[2]["message"].tool_call_id)

    def test_continue_stops_at_limit_without_extra_model_or_tool_calls(self):
        events, http = self.run_loop([
            self.packet(name="multiply", args={"a": 3, "b": 4}), self.packet(self.pending),
            self.packet(name="add", args={"a": 12, "b": 5}), self.packet(self.pending)])
        self.assertEqual(http.call_count, 4)
        self.assertEqual(len(self.invocations), 2)
        self.assertFalse(events[-1]["task_complete"])
        self.assertEqual(events[-1]["stop_reason"], "max_iterations")
        self.assertIn("2轮", events[-1]["full_response"])


    def test_single_rag_preserves_citations_when_action_keeps_whole_question(self):
        """单一RAG任务传递完整原问题，直通实际答案，不二次生成或丢失引用。"""
        question = "请根据知识库查询代号，并引用文档名和行号"
        context = {"observations": [{"name": "knowledge_base_search", "status": "success",
            "args": {"question": question}, "result": {"generation_mode": "grounded", "status": "answered",
                "citations": [{"id": 1}], "answer": "代号WINCHECK[参考文档1：说明.txt；行1–3]。"}}]}
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.packet(self.finished)).encode())) as http:
            result = observe(question, [knowledge_base_search], context)
        self.assertEqual(result["answer"], context["observations"][0]["result"]["answer"])
        self.assertTrue(result["task_complete"])
        http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
