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
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from src.agent.react_loop import act, observe, run_react, think
from src.agent.tools import AVAILABLE_TOOLS
from src.agent.tools import current_time, web_search
from src.agent.router import execute_calls, parallel_limit, route_question
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

    def test_clear_question_types_route_only_to_registered_tools_without_model(self):
        cases = [("现在几点，返回当前时间", "current_time"), ("current time please", "current_time"),
                 ("提取问题的关键词", "keyword_extract"), ("生成论文的结构化摘要", "paper_summary"),
                 ("查询论文元信息", "paper_metadata"), ("对比两篇论文", "paper_compare"),
                 ("这篇论文使用了什么数据集？", "knowledge_base_search"), ("论文A的实验数据集是什么？", "knowledge_base_search")]
        with patch("src.agent.react_loop.urlopen") as http:
            for question, name in cases:
                with self.subTest(question=question):
                    plan = route_question(question, AVAILABLE_TOOLS)
                    self.assertEqual(plan["tool_name"], name)
                    self.assertEqual(plan["usage"], {"prompt_eval_count": 0, "eval_count": 0})
                    self.assertIsNone(plan["model"])
        http.assert_not_called()

    def test_general_concept_routes_to_answer_but_paper_question_uses_knowledge(self):
        self.assertEqual(route_question("什么是深度学习？", AVAILABLE_TOOLS)["next_step"], "answer")
        self.assertEqual(route_question("这篇论文中的深度学习是什么？", AVAILABLE_TOOLS)["tool_name"], "knowledge_base_search")

    def test_ambiguous_dependent_negated_and_existing_state_fall_back(self):
        for question in ("当前时间并提取关键词", "先提取关键词，再用关键词查询知识库", "不要调用current_time", "帮我处理一下"):
            with self.subTest(question=question):
                self.assertIsNone(route_question(question, AVAILABLE_TOOLS))
        for context in ({"observations": [{"status": "success"}]}, {"history": ["旧问题"]}, {"context": "论文资料"}):
            self.assertIsNone(route_question("什么是深度学习？", AVAILABLE_TOOLS, context))

    def test_missing_or_disabled_tools_fall_back_without_fake_registry(self):
        self.assertEqual(route_question("联网搜索论文", AVAILABLE_TOOLS)["unavailable_tool"], "web_search")
        self.assertIsNone(route_question("返回当前时间", []))
        self.assertIsNone(route_question("3.14乘以2.56", [item for item in AVAILABLE_TOOLS if item.name != "calculator"]))
        self.assertEqual(route_question("联网搜索论文", [web_search])["tool_name"], "web_search")

    def test_explicit_tool_names_and_two_papers_use_same_tool_batch(self):
        plan = route_question("请调用paper_metadata，分别读取两篇论文：" + "a" * 64 + "和" + "b" * 64, AVAILABLE_TOOLS)
        self.assertEqual(plan["tool_name"], "paper_metadata")
        self.assertEqual(plan["parallel_tools"], ["paper_metadata"])
        self.assertEqual(route_question("请调用keyword_extract提取问题主题", AVAILABLE_TOOLS)["tool_name"], "keyword_extract")
        self.assertIsNone(route_question("请调用paper_metadata和paper_summary", AVAILABLE_TOOLS))

    def test_calculator_rule_requires_actual_registered_tool(self):
        @tool("calculator")
        def multiply(a: float, b: float) -> float:
            """开发测试用的实际乘法工具，不计入生产八工具。"""
            return a * b
        self.assertEqual(route_question("3.14乘以2.56", [multiply])["tool_name"], "calculator")
        self.assertIsNone(route_question("3.14乘以2.56", []))

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

    def test_rule_only_applies_to_first_round(self):
        pending = {**self.finished, "decision": "continue", "task_complete": False, "answer": ""}
        response = {**self.response, "message": {"tool_calls": [{"function": {"name": "current_time", "arguments": {}}}]}}
        with patch("src.agent.react_loop.think", return_value={"thought": "已有时间，可回答。", "next_step": "answer", "tool_name": None}) as think_mock, \
                patch("src.agent.react_loop.urlopen", side_effect=[BytesIO(json.dumps(x).encode()) for x in
                    (response, self.packet(pending), self.packet(self.finished))]):
            events = list(run_react("返回当前时间", [current_time]))
        self.assertEqual(think_mock.call_count, 1)
        self.assertEqual(len(events[-1]["context"]["observations"]), 1)

    def test_thought_accepts_independent_names_list_and_schema_limit(self):
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.packet(self.plan)).encode())) as http:
            result = think("独立读取3和5", self.tools)
        self.assertEqual(result["parallel_tools"], ["read_number"])
        choice = json.loads(http.call_args.args[0].data)["format"]["properties"]["parallel_tools"]
        self.assertEqual(choice["maxItems"], 2)
        self.assertNotIn("uniqueItems", choice)

    def test_same_tool_two_inputs_are_accepted_without_duplicate_schema(self):
        plan = {**self.plan, "parallel_tools": ["read_number", "read_number"]}
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.packet(plan)).encode())):
            self.assertEqual(think("分别读取3和5", self.tools)["parallel_tools"], plan["parallel_tools"])
        events, http = self.action(plan=plan)
        self.assertEqual(sorted(self.invocations), [3, 5])
        self.assertEqual(len(json.loads(http.call_args.args[0].data)["tools"]), 1)
        self.assertEqual(sum(e["type"] == "tool_result" for e in events), 2)
        response = {**self.response, "message": {"tool_calls": self.response["message"]["tool_calls"][:1]}}
        events, _ = self.action(response=response, plan=plan)
        self.assertEqual(events[-1]["type"], "error")

    def test_thought_rejects_empty_duplicate_unknown_or_excessive_batch(self):
        for names in (None, "read_number", ["read_number"] * 3, ["unknown"], [1], ["read_number", "unknown", "other"]):
            with self.subTest(names=names), patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(
                    self.packet({**self.plan, "parallel_tools": names})).encode())), self.assertRaises(RuntimeError):
                think("独立读取", self.tools)

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

    def test_serial_path_and_single_call_keep_true_results_and_order(self):
        results = list(execute_calls(self.calls, self.tools))
        self.assertEqual(self.invocations, [3, 5])
        self.assertEqual([item["result"] for item in results], [6, 10])
        self.assertTrue(all(item["execution_mode"] == "serial" for item in results))
        self.assertEqual(list(execute_calls(self.calls[:1], self.tools, parallel=True))[0]["execution_mode"], "serial")

    def test_individual_failure_preserves_other_independent_result(self):
        calls = deepcopy(self.calls)
        calls[0]["args"]["value"] = -1
        results = list(execute_calls(calls, self.tools, parallel=True))
        self.assertEqual([item["status"] for item in results], ["error", "success"])
        self.assertIn("输入不能为负数", results[0]["error"])
        self.assertEqual(results[1]["result"], 10)
        self.assertCountEqual(self.invocations, [-1, 5])

    def test_batch_limit_duplicate_ids_and_invalid_config_are_rejected(self):
        for calls in ([], [*self.calls, {**self.calls[0], "call_id": "call-c"}], [self.calls[0]] * 2):
            with self.subTest(calls=calls), self.assertRaises(ValueError):
                list(execute_calls(calls, self.tools, parallel=True))
        for value in (0, -1, True, "2"):
            self.config["agent"]["max_parallel_calls"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                parallel_limit()
        self.assertEqual(self.invocations, [])

    def test_native_batch_has_one_ai_message_two_linked_results_and_one_usage(self):
        events, http = self.action()
        self.assertEqual([e["type"] for e in events], ["tool_call", "tool_call", "tool_result", "tool_result"])
        self.assertEqual([e["result"] for e in events[2:]], [6, 10])
        self.assertEqual(len(events[0]["message"].tool_calls), 2)
        self.assertNotIn("message", events[1])
        self.assertEqual(sum(e.get("usage", {}).get("eval_count", 0) for e in events), 60)
        self.assertEqual(http.call_count, 1)
        for call, result in zip(events[:2], events[2:]):
            self.assertEqual(call["call_id"], result["message"].tool_call_id)
            self.assertEqual(call["args"], result["args"])

    def test_batch_supports_different_selected_tools_only(self):
        @tool
        def other(value: int) -> int:
            """另一个独立输入工具。"""
            return value + 1
        response = deepcopy(self.response)
        response["message"]["tool_calls"][1]["function"]["name"] = "other"
        events, http = self.action(response, {**self.plan, "parallel_tools": ["read_number", "other"]}, [*self.tools, other, current_time])
        self.assertEqual([e["result"] for e in events[2:]], [6, 6])
        self.assertEqual([spec["function"]["name"] for spec in json.loads(http.call_args.args[0].data)["tools"]], ["read_number", "other"])

    def test_invalid_batch_protocol_rejected_before_any_execution(self):
        for mode in ("unknown", "duplicate", "excessive", "unfinished"):
            response = deepcopy(self.response)
            if mode == "unknown":
                response["message"]["tool_calls"][1]["function"]["name"] = "other"
            elif mode == "duplicate":
                response["message"]["tool_calls"][1] = deepcopy(response["message"]["tool_calls"][0])
            elif mode == "excessive":
                response["message"]["tool_calls"].append(deepcopy(response["message"]["tool_calls"][0]))
            else:
                response["done_reason"] = "length"
            events, _ = self.action(response)
            with self.subTest(mode=mode):
                self.assertEqual([e["type"] for e in events], ["error"])
        self.assertEqual(self.invocations, [])

    def test_batch_cannot_omit_selected_tool_or_mismatch_primary_name(self):
        plan = {**self.plan, "parallel_tools": ["read_number", "current_time"]}
        packet = deepcopy(self.response)
        packet["message"]["tool_calls"] = packet["message"]["tool_calls"][:1]
        events, _ = self.action(packet, plan, [*self.tools, current_time])
        self.assertEqual([event["type"] for event in events], ["error"])
        self.assertIn("完整", events[0]["message"])
        for bad in ({**self.plan, "tool_name": "current_time"}, {**self.plan, "next_step": "answer", "tool_name": None}):
            with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.packet(bad)).encode())), \
                    self.subTest(plan=bad), self.assertRaises(RuntimeError):
                think("读取独立输入", [*self.tools, current_time])
        self.assertEqual(self.invocations, [])

    def test_closing_during_batch_call_events_starts_no_tools(self):
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.response).encode())):
            stream = act("独立输入", self.plan, self.tools)
            next(stream)
            next(stream)
            stream.close()
        self.assertEqual(self.invocations, [])

    def test_error_or_confirmation_anywhere_in_current_batch_cannot_complete(self):
        for result in ({"status": "error", "result": None}, {"status": "success", "result": {"status": "needs_confirmation"}},
                       {"status": "success", "result": {"status": "insufficient_evidence"}}):
            context = {"observations": [result, {"status": "success", "result": "最后一个成功"}]}
            messages = [ToolMessage(content="第一条", tool_call_id="a"), ToolMessage(content="第二条", tool_call_id="b")]
            with self.subTest(result=result), patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.packet(self.finished)).encode())), \
                    self.assertRaises(RuntimeError):
                observe("完成两件事", self.tools, context, messages)

    def test_loop_observes_entire_batch_and_preserves_native_input_order(self):
        with patch("src.agent.react_loop.urlopen", side_effect=[BytesIO(json.dumps(x).encode()) for x in
                (self.packet(self.plan), self.response, self.packet(self.finished))]) as http:
            events = list(run_react("处理独立输入3和5。", self.tools))
        self.assertEqual([item["result"] for item in events[-1]["context"]["observations"]], [6, 10])
        self.assertTrue(events[-1]["task_complete"])
        native = json.loads(http.call_args_list[-1].args[0].data)["messages"]
        self.assertEqual([m["role"] for m in native], ["system", "user", "assistant", "tool", "tool"])
        self.assertEqual(len(native[2]["tool_calls"]), 2)
        self.assertEqual([m["content"] for m in native[-2:]], ["6", "10"])

    def test_answer_plan_finishes_or_reports_insufficient_information_without_empty_loop(self):
        plan = {"thought": "已有结果可给出答复。", "next_step": "answer", "tool_name": None}
        pending = {**self.finished, "decision": "continue", "task_complete": False, "answer": ""}
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.packet(pending)).encode())) as http, \
                self.assertRaisesRegex(RuntimeError, "不能继续空转"):
            observe("处理任务", self.tools, thought=plan)
        schema = json.loads(http.call_args.args[0].data)["format"]
        self.assertEqual(schema["properties"]["decision"], {"const": "finish"})
        incomplete = {**self.finished, "task_complete": False, "answer": "缺少论文，无法完成。"}
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.packet(incomplete)).encode())):
            result = observe("处理任务", self.tools, thought=plan)
        self.assertFalse(result["task_complete"])


if __name__ == "__main__":
    unittest.main()
