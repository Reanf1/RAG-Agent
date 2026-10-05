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
from urllib.error import URLError
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

    def test_two_real_tools_feed_next_round_and_native_observation_messages(self):
        context = {"source": "原始资料", "observations": []}
        original = deepcopy(context)
        events, http = self.run_loop([
            self.plan("multiply"), self.packet(name="multiply", args={"a": 3, "b": 4}), self.packet(self.pending),
            self.plan("add"), self.packet(name="add", args={"a": 12, "b": 5}), self.packet(self.finished)], context)
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
        native = json.loads(http.call_args_list[2].args[0].data)["messages"]
        self.assertEqual([m["role"] for m in native], ["system", "user", "assistant", "tool"])
        self.assertEqual(native[2]["tool_calls"][0]["function"], {"name": "multiply", "arguments": {"a": 3, "b": 4}})
        self.assertEqual(native[3]["tool_name"], "multiply")
        self.assertEqual(native[3]["content"], "12.0")
        second_state = json.loads(json.loads(http.call_args_list[3].args[0].data)["messages"][1]["content"])["context"]
        self.assertEqual(second_state["observations"][0]["result"], 12)
        self.assertEqual(second_state["last_observation"]["decision"], "continue")
        self.assertEqual(events[1]["message"].tool_calls[0]["id"], events[2]["message"].tool_call_id)

    def test_continue_stops_at_limit_without_extra_model_or_tool_calls(self):
        events, http = self.run_loop([
            self.plan("multiply"), self.packet(name="multiply", args={"a": 3, "b": 4}), self.packet(self.pending),
            self.plan("add"), self.packet(name="add", args={"a": 12, "b": 5}), self.packet(self.pending)])
        self.assertEqual(http.call_count, 6)
        self.assertEqual(len(self.invocations), 2)
        self.assertFalse(events[-1]["task_complete"])
        self.assertEqual(events[-1]["stop_reason"], "max_iterations")
        self.assertIn("2轮", events[-1]["full_response"])

    def test_direct_answer_finishes_without_tool_calling_request(self):
        events, http = self.run_loop([self.plan(), self.packet(self.finished)])
        self.assertEqual([e["type"] for e in events], ["thought", "action_skipped", "observation", "done"])
        self.assertEqual(http.call_count, 2)
        self.assertEqual(self.invocations, [])
        self.assertTrue(events[-1]["task_complete"])
        self.assertEqual(events[-1]["iterations"], 1)

    def test_missing_information_can_finish_without_claiming_completion(self):
        decision = {**self.finished, "task_complete": False, "answer": "请提供论文内容，当前无法回答。"}
        events, _ = self.run_loop([self.plan(), self.packet(decision)])
        self.assertEqual(events[-1]["stop_reason"], "incomplete")
        self.assertFalse(events[-1]["task_complete"])
        self.assertIn("无法回答", events[-1]["full_response"])

    def test_empty_library_notice_survives_observation_rewriting(self):
        """复现W05：工具已声明空库，Observation却只返回纯模型正文。"""
        notice = "当前知识库中未找到相关文档。以下为纯模型回答，没有知识库文献依据。"
        context = {"observations": [{"name": "knowledge_base_search", "status": "success",
            "result": {"generation_mode": "empty", "status": "answered", "notice": notice,
                       "answer": notice + "\n\n位置编码表示顺序。"}}]}
        decision = {**self.finished, "answer": "位置编码表示顺序。"}
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.packet(decision)).encode())):
            result = observe("请根据知识库解释位置编码", [], context)
        self.assertTrue(result["answer"].startswith(notice))
        self.assertEqual(result["answer"].count(notice), 1)

    def test_retrieved_text_without_citation_is_not_rewritten_as_empty_library(self):
        """复现W07：无有效引用不等于检索无文档，保留真实工具状态。"""
        question = "根据知识库文档" + "a" * 64 + "回答复测代号是什么？"
        context = {"observations": [{"name": "knowledge_base_search", "status": "success",
            "args": {"question": "复测代号是什么？"}, "result": {
                "generation_mode": "grounded", "status": "insufficient_evidence", "citations": [],
                "sources": ["说明.txt"], "answer": "复测代号是WINCHECK20261005。",
                "warnings": ["回答未提供有效文献引用，不能作为已溯源答案。"]}}]}
        decision = {**self.finished, "task_complete": False, "answer": "当前知识库中未找到相关文档。"}
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.packet(decision)).encode())) as http:
            result = observe(question, [knowledge_base_search], context)
        self.assertFalse(result["task_complete"])
        self.assertIn("WINCHECK20261005", result["answer"])
        self.assertIn("未提供有效文献引用", result["answer"])
        self.assertNotIn("未找到相关文档", result["answer"])
        http.assert_not_called()

    def test_single_rag_preserves_citations_after_action_shortens_question(self):
        """单一RAG任务的查询改写不能导致Observation再次生成和丢失引用。"""
        context = {"observations": [{"name": "knowledge_base_search", "status": "success",
            "args": {"question": "代号？"}, "result": {"generation_mode": "grounded", "status": "answered",
                "citations": [{"id": 1}], "answer": "代号WINCHECK[参考文档1：说明.txt；行1–3]。"}}]}
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.packet(self.finished)).encode())) as http:
            result = observe("请根据知识库查询代号，并引用文档名和行号", [knowledge_base_search], context)
        self.assertEqual(result["answer"], context["observations"][0]["result"]["answer"])
        self.assertTrue(result["task_complete"])
        http.assert_not_called()

    def test_requested_missing_doi_is_explicit_in_final_metadata_answer(self):
        """复现W02：工具已返回DOI缺失，最终回答不得省略用户请求的字段。"""
        context = {"observations": [{"name": "paper_metadata", "status": "success",
            "result": {"title": "DETR", "doi": None, "missing_fields": ["doi"]}}]}
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.packet(self.finished)).encode())):
            result = observe("提取标题、作者、年份、摘要、DOI", [], context)
        self.assertIn("DOI：原文未提供", result["answer"])

    def test_tool_failure_is_observed_and_stops_honestly(self):
        failed = {"observation": "除数为零，工具失败。", "decision": "finish",
                  "task_complete": False, "answer": "除零无法计算，请修改除数。"}
        events, http = self.run_loop([self.plan("divide"), self.packet(name="divide", args={"a": 1, "b": 0}),
                                      self.packet(failed)])
        self.assertEqual(self.invocations, [("divide", 1, 0)])
        self.assertEqual(events[2]["status"], "error")
        state = json.loads(json.loads(http.call_args_list[2].args[0].data)["messages"][1]["content"])["context"]
        self.assertIn("ZeroDivisionError", state["observations"][0]["error"])
        self.assertIsNone(state["observations"][0]["result"])
        self.assertEqual(events[-1]["stop_reason"], "incomplete")

    def test_failed_tool_cannot_be_marked_successful(self):
        events, _ = self.run_loop([self.plan("divide"), self.packet(name="divide", args={"a": 1, "b": 0}),
                                   self.packet(self.finished)])
        self.assertEqual([e["type"] for e in events[-2:]], ["error", "done"])
        self.assertFalse(events[-1]["task_complete"])
        self.assertEqual(events[-1]["stop_reason"], "error")
        self.assertEqual(len(events[-1]["context"]["observations"]), 1)

    def test_tool_error_can_continue_to_a_real_alternative(self):
        events, _ = self.run_loop([
            self.plan("divide"), self.packet(name="divide", args={"a": 1, "b": 0}), self.packet(self.pending),
            self.plan("add"), self.packet(name="add", args={"a": 12, "b": 5}), self.packet(self.finished)])
        self.assertTrue(events[-1]["task_complete"])
        self.assertEqual([o["status"] for o in events[-1]["context"]["observations"]], ["error", "success"])

    def test_model_errors_stop_in_each_stage_and_preserve_actual_results(self):
        first = [self.plan("multiply"), self.packet(name="multiply", args={"a": 3, "b": 4})]
        for responses in ([URLError("Thought断开")], [first[0], URLError("Action断开")],
                          [*first, URLError("Observation断开")]):
            with self.subTest(stage=len(responses)):
                events, http = self.run_loop(responses)
                self.assertEqual(http.call_count, len(responses))
                self.assertEqual(events[-1]["stop_reason"], "error")
                self.assertFalse(events[-1]["task_complete"])
                self.assertEqual(sum(e["type"] == "error" for e in events), 1)
                self.assertEqual(len(events[-1]["context"]["observations"]), int(len(responses) == 3))

    def test_invalid_limits_and_context_stop_before_model_calls(self):
        for limit in (0, -1, 1.5, True, "2"):
            self.config["agent"]["max_iterations"] = limit
            events, http = self.run_loop([])
            self.assertEqual(events[-1]["iterations"], 0)
            self.assertFalse(events[-1]["task_complete"])
            http.assert_not_called()
        self.config["agent"]["max_iterations"] = 2
        for context in ([], {"observations": {}}, {"observations": ["非法记录"]}):
            events, http = self.run_loop([], context)
            self.assertEqual(events[-1]["stop_reason"], "error")
            http.assert_not_called()

    def test_closing_loop_and_mutating_events_do_not_execute_or_change_state(self):
        responses = [self.plan("multiply"), self.packet(name="multiply", args={"a": 3, "b": 4}),
                     self.packet({**self.finished, "answer": "12"})]
        with patch("src.agent.react_loop.urlopen", side_effect=self.http_responses(responses)):
            loop = run_react("3乘4", self.tools)
            next(loop)["tool_name"] = "divide"  # 外部展示事件不能改变内部计划。
            call = next(loop)
            self.assertEqual(call["name"], "multiply")
            call["args"]["a"] = 999
            result = next(loop)
            result["result"] = 999
            rest = list(loop)
        self.assertEqual(rest[-1]["context"]["observations"][0]["result"], 12)
        self.invocations.clear()
        with patch("src.agent.react_loop.urlopen", side_effect=self.http_responses(responses)) as http:
            loop = run_react("3乘4", self.tools)
            next(loop)
            next(loop)
            loop.close()
        self.assertEqual(http.call_count, 2)
        self.assertEqual(self.invocations, [])

    def test_observation_rejects_inconsistent_decisions(self):
        decisions = [{**self.finished, "task_complete": 1}, {**self.finished, "decision": "other"},
                     {**self.finished, "answer": " "}, {**self.pending, "task_complete": True},
                     {**self.pending, "answer": "提前回答"}, {**self.finished, "answer": None},
                     {**self.finished, "observation": "长" * 201}, {**self.finished, "observation": " "},
                     {**self.finished, "extra": 1}, [], {}]
        for decision in decisions:
            with self.subTest(decision=decision), \
                    patch("src.agent.react_loop.urlopen", return_value=self.http_responses([self.packet(decision)])[0]), \
                    self.assertRaises(RuntimeError):
                observe("问题", self.tools)

    def test_observation_rejects_partial_malformed_and_service_responses(self):
        valid = self.packet(self.finished)
        responses = [{**valid, "done": False}, {**valid, "done_reason": "length"},
                     {**valid, "model": None}, {**valid, "message": []}, {"error": "加载失败"},
                     {**valid, "message": {"content": None}}, {**valid, "message": {"content": "不是JSON"}}]
        for response in responses:
            with self.subTest(response=response), \
                    patch("src.agent.react_loop.urlopen", return_value=self.http_responses([response])[0]), \
                    self.assertRaises(RuntimeError):
                observe("问题")

    def test_observation_unknown_usage_and_boundary_description(self):
        response = self.packet({**self.finished, "observation": "短" * 200})
        del response["prompt_eval_count"], response["eval_count"]
        with patch("src.agent.react_loop.urlopen", return_value=self.http_responses([response])[0]):
            result = observe("问题")
        self.assertEqual(result["usage"], {"prompt_eval_count": None, "eval_count": None})
        self.assertEqual(len(result["observation"]), 200)

    def test_observation_validates_state_before_call_and_never_uses_cloud(self):
        with patch("src.agent.react_loop.urlopen") as http:
            with self.assertRaises(ValueError):
                observe("问题", context={"observations": ["非法状态"]})
            self.config["llm"]["base_url"] = "http://example.com"
            with patch("src.agent.react_loop.load_config", return_value=self.config), self.assertRaises(ValueError):
                observe("问题")
            http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
