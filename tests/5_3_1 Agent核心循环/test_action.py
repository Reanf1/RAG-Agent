"""5.3.1 Agent核心循环：TestAction。"""

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
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from src.agent.react_loop import act
from src.agent.tools import execute_tool
from src.utils.config import load_config
from tests.helpers import isolated_agent_logs


def setUpModule():
    """保持既有 Agent 测试的模块级日志隔离。"""
    global _log_context
    _log_context = isolated_agent_logs()
    _log_context.__enter__()


def tearDownModule():
    _log_context.__exit__(None, None, None)


class TestAction(unittest.TestCase):
    """真实工具执行与参数校验；模型调用使用原生Function Calling响应样例。"""

    def setUp(self):
        self.invocations = []

        @tool
        def multiply(a: float, b: float) -> float:
            """计算两个数的乘积。"""
            self.invocations.append((a, b))
            return a * b

        self.tools = [multiply]
        self.thought = {"type": "thought", "thought": "用乘法工具取得准确结果。", "next_step": "tool", "tool_name": "multiply"}
        self.response = {"model": "qwen2.5:7b", "done": True, "done_reason": "stop",
                         "message": {"tool_calls": [{"function": {"name": "multiply", "arguments": {"a": 3.14, "b": 2.56}}}]},
                         "prompt_eval_count": 300, "eval_count": 30}

    def run_action(self, response=None, thought=None, tools=None, context=None):
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.response if response is None else response).encode())):
            return list(act("3.14乘以2.56", self.thought if thought is None else thought,
                            self.tools if tools is None else tools, context))

    def test_native_call_executes_selected_tool_and_correlates_messages(self):
        context = {"observations": [{"tool_name": "other", "result": "旧结果"}]}
        snapshot = deepcopy((context, self.thought))
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.response).encode())) as http:
            events = list(act("3.14乘以2.56", self.thought, self.tools, context))
        self.assertEqual([item["type"] for item in events], ["tool_call", "tool_result"])
        call, result = events
        self.assertEqual(self.invocations, [(3.14, 2.56)])
        self.assertAlmostEqual(result["result"], 8.0384)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["error"], "")
        self.assertIsInstance(call["message"], AIMessage)
        self.assertIsInstance(result["message"], ToolMessage)
        self.assertEqual(call["call_id"], result["message"].tool_call_id)
        self.assertEqual(call["message"].tool_calls[0]["id"], result["call_id"])
        self.assertEqual(call["usage"], {"prompt_eval_count": 300, "eval_count": 30})
        self.assertGreaterEqual(result["elapsed_seconds"], 0)
        payload = json.loads(http.call_args.args[0].data)
        self.assertNotIn("format", payload)
        self.assertEqual([item["function"]["name"] for item in payload["tools"]], ["multiply"])
        self.assertEqual(json.loads(payload["messages"][1]["content"])["context"], context)
        self.assertEqual((context, self.thought), snapshot)

    def test_answer_plan_skips_model_and_tool_execution(self):
        with patch("src.agent.react_loop.urlopen") as http:
            events = list(act("什么是过拟合？", {"next_step": "answer", "tool_name": None}, []))
        self.assertEqual([item["type"] for item in events], ["action_skipped"])
        http.assert_not_called()
        self.assertEqual(self.invocations, [])

    def test_filename_preflight_uses_declared_empty_paper_list_arguments(self):
        """复现W01：模型曾把文件名塞入无参数文献列表的query字段。"""
        @tool
        def paper_list() -> dict:
            """列出全部文献，无查询参数。"""
            self.invocations.append("paper_list")
            return {"papers": [], "total": 0}

        response = deepcopy(self.response)
        response["message"]["tool_calls"] = [{"function": {
            "name": "paper_list", "arguments": {"query": "Windows验收_ViT.pdf"}}}]
        thought = {**self.thought, "tool_name": "paper_list"}
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(response).encode())) as http:
            events = list(act("请检索Windows验收_ViT.pdf的位置编码", thought, [paper_list]))
        self.assertEqual(events[0]["args"], {})
        self.assertEqual(events[-1]["status"], "success")
        self.assertEqual(self.invocations, ["paper_list"])
        http.assert_not_called()
        self.assertEqual(events[0]["usage"], {"prompt_eval_count": 0, "eval_count": 0})

    def test_keyword_document_filename_preflight_and_alias(self):
        """文档关键词入口先查列表，再将唯一文件名映射为真实ID。"""
        from src.agent.react_loop import run_react
        @tool
        def paper_list() -> dict:
            """提供实际上传的文档指纹。"""
            return {"papers": [{"doc_id": "a" * 64, "source_file": "目标.md"}], "total": 1}
        @tool
        def keyword_extract(doc_id: str) -> dict:
            """核验最终收到的文档ID。"""
            self.invocations.append(doc_id)
            return {"keywords": ["测试"]}
        call = {**self.response, "message": {"tool_calls": [{"function": {
            "name": "keyword_extract", "arguments": {"doc_id": "目标.md"}}}]}}
        done = {**self.response, "message": {"content": json.dumps({"observation": "关键词已提取", "decision": "finish",
                "task_complete": True, "answer": "关键词为测试"})}}
        with patch("src.agent.react_loop.urlopen", side_effect=[BytesIO(json.dumps(item).encode()) for item in (call, done)]):
            events = list(run_react("从目标.md提取关键词", [paper_list, keyword_extract]))
        self.assertEqual([e["name"] for e in events if e["type"] == "tool_call"], ["paper_list", "keyword_extract"])
        self.assertEqual(self.invocations, ["a" * 64])
        self.assertTrue(events[-1]["task_complete"])

    def test_current_paper_id_overrides_model_choice_from_old_history(self):
        """复现W09：最新问题指定ViT，模型Action却填了历史DETR的真实ID。"""
        @tool
        def paper_summary(doc_id: str) -> dict:
            """记录摘要工具实际收到的目标。"""
            self.invocations.append(doc_id)
            return {"doc_id": doc_id}

        current, previous = "a" * 64, "b" * 64
        response = deepcopy(self.response)
        response["message"]["tool_calls"] = [{"function": {
            "name": "paper_summary", "arguments": {"doc_id": previous}}}]
        context = {"history": [{"role": "human", "content": "总结论文" + previous}]}
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(response).encode())):
            events = list(act("针对论文" + current + "生成结构化摘要",
                              {**self.thought, "tool_name": "paper_summary"}, [paper_summary], context))
        self.assertEqual(self.invocations, [current])
        self.assertEqual(events[0]["args"]["doc_id"], current)

    def test_text_keyword_input_is_not_changed_to_conflicting_document_input(self):
        """关键词处理已给定文本时，即使问题提到ID，也不能自动再补doc_id。"""
        @tool
        def keyword_extract(text: str | None = None, doc_id: str | None = None) -> dict:
            """记录互斥输入，不读取论文。"""
            self.invocations.append((text, doc_id))
            if (text is None) == (doc_id is None):
                raise ValueError("text和doc_id必须且只能提供一个")
            return {"keywords": ["Transformer"]}
        response = {**self.response, "message": {"tool_calls": [{"function": {
            "name": "keyword_extract", "arguments": {"text": "Transformer用于科研"}}}]}}
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(response).encode())):
            events = list(act("论文" + "a" * 64 + "；请提取给定文本Transformer用于科研的关键词",
                              {**self.thought, "tool_name": "keyword_extract"}, [keyword_extract]))
        self.assertEqual(self.invocations, [("Transformer用于科研", None)])
        self.assertEqual(events[-1]["status"], "success")

    def test_current_document_filter_cannot_be_omitted_by_action(self):
        """最新问题的单个完整ID必须成为检索过滤，不扩展为全库查询。"""
        identifier = "c" * 64
        question = "查询知识库文档" + identifier + "的代号"
        events, _ = self.knowledge_action({"question": "代号是什么？"}, question,
                                         {"history": [{"role": "ai", "content": "另一篇论文使用1D编码"}]})
        self.assertEqual(events[0]["args"]["doc_id"], identifier)

    def knowledge_action(self, args, question="已上传attention.pdf的编码器有多少层？", context=None, route=None):
        """模拟模型返回参数，真实执行小工具核验Action契约，不作为RAG质量证据。"""
        @tool
        def knowledge_base_search(question: str, doc_id: str | None = None) -> str:
            """记录知识库工具实际收到的问题和可选论文指纹。"""
            self.invocations.append((question, doc_id))
            return "已执行"

        response = deepcopy(self.response)
        response["message"]["tool_calls"] = [{"function": {"name": "knowledge_base_search", "arguments": args}}]
        thought = {**self.thought, "tool_name": "knowledge_base_search", "route": route}
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(response).encode())) as http:
            events = list(act(question, thought, [knowledge_base_search], context))
        if http.call_args is None:
            self.assertEqual(route, "rule")
            http.assert_not_called()
            return events, None
        payload = json.loads(http.call_args.args[0].data)
        described = json.loads(payload["messages"][0]["content"].split("【可用工具描述】\n")[1].splitlines()[0])
        self.assertEqual(described["available_tools"], [{key: spec["function"][key] for key in ("name", "description")} for spec in payload["tools"]])
        return events, payload["tools"][0]["function"]["parameters"]

    def test_filename_only_rule_search_passes_question_without_model_or_invented_id(self):
        question = "已上传attention.pdf的编码器有多少层？"
        events, _ = self.knowledge_action({"question": question}, question,
                                         {"session_id": "a" * 32}, route="rule")
        self.assertEqual(self.invocations, [(question, None)])
        self.assertEqual(events[-1]["status"], "success")
        self.assertEqual(events[0]["usage"], {"prompt_eval_count": 0, "eval_count": 0})
        self.assertIsNone(events[0]["model"])

    def test_provided_fingerprint_in_question_or_tool_context_keeps_document_filter(self):
        identifier = "b" * 64
        for question, context, route in (("查询这篇论文" + identifier, None, "rule"),
                                         ("查询attention.pdf", {"observations": [{"result": {"papers": [{"doc_id": identifier}]}}]}, None)):
            with self.subTest(context=context):
                events, parameters = self.knowledge_action({"question": question, "doc_id": identifier}, question, context, route)
                self.assertIn("doc_id", parameters["properties"])
                self.assertEqual(events[-1]["status"], "success")
                self.assertEqual(self.invocations[-1], (question, identifier))

    def test_unprovided_or_invalid_fingerprint_never_executes_tool(self):
        for identifier in ("a" * 32, "b" * 64, ["b" * 64]):
            with self.subTest(identifier=identifier):
                events, _ = self.knowledge_action({"question": "查询论文", "doc_id": identifier})
                self.assertEqual([event["type"] for event in events], ["error"])
                self.assertIn("不能编造论文指纹", events[0]["message"])
        self.assertEqual(self.invocations, [])

    def test_only_selected_tool_is_exposed_even_when_other_tools_are_available(self):
        @tool
        def other() -> str:
            """另一个不应调用的工具。"""
            raise AssertionError("不能执行未选择的工具")

        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.response).encode())) as http:
            list(act("3.14乘以2.56", self.thought, [other, *self.tools]))
        self.assertEqual(len(json.loads(http.call_args.args[0].data)["tools"]), 1)
        system = json.loads(http.call_args.args[0].data)["messages"][0]["content"]
        described = json.loads(system.split("【可用工具描述】\n")[1].splitlines()[0])["available_tools"]
        self.assertEqual([item["name"] for item in described], ["multiply"])
        self.assertEqual(len(self.invocations), 1)

    def test_invalid_thought_and_duplicate_tools_do_not_call_model(self):
        cases = [([], self.tools), ({"next_step": "execute"}, self.tools),
                 ({"next_step": "answer", "tool_name": "multiply"}, self.tools),
                 ({**self.thought, "tool_name": "unknown"}, self.tools), (self.thought, self.tools * 2)]
        for thought, tools in cases:
            with self.subTest(thought=thought), patch("src.agent.react_loop.urlopen") as http:
                events = list(act("问题", thought, tools))
                self.assertEqual(events[0]["type"], "error")
                http.assert_not_called()
        self.assertEqual(self.invocations, [])

    def test_empty_multiple_wrong_name_and_non_dict_arguments_do_not_execute(self):
        valid = deepcopy(self.response["message"]["tool_calls"][0])
        calls = [None, [], [valid, valid], ["非法调用"], [{"function": []}],
                 [{"function": {"name": "other", "arguments": {}}}],
                 [{"function": {"name": "multiply", "arguments": '{"a":3,"b":4}'}}]]
        for item in calls:
            with self.subTest(calls=item):
                events = self.run_action({**self.response, "message": {"tool_calls": item}})
                self.assertEqual([e["type"] for e in events], ["error"])
        self.assertEqual(self.invocations, [])

    def test_required_type_and_extra_argument_errors_are_tool_results_without_side_effects(self):
        for args in ({}, {"a": "非数字", "b": 2}, {"a": 1, "b": 2, "extra": 3}, {"a": float("nan"), "b": 2}):
            with self.subTest(args=args):
                response = deepcopy(self.response)
                response["message"]["tool_calls"][0]["function"]["arguments"] = args
                events = self.run_action(response)
                self.assertEqual(events[-1]["status"], "error")
                self.assertEqual(events[-1]["message"].status, "error")
                self.assertIsNone(events[-1]["result"])
        self.assertEqual(self.invocations, [])

    def test_tool_exception_is_preserved_and_not_retried(self):
        @tool
        def divide(a: float, b: float) -> float:
            """两数相除，除零应明确失败。"""
            self.invocations.append((a, b))
            return a / b

        result = execute_tool("divide", {"a": 1, "b": 0}, [divide], "call-failure")
        self.assertEqual(result["status"], "error")
        self.assertIn("ZeroDivisionError", result["error"])
        self.assertEqual(result["message"].tool_call_id, "call-failure")
        self.assertEqual(self.invocations, [(1, 0)])

    def test_no_argument_and_optional_default_tools_execute(self):
        @tool
        def ping() -> str:
            """返回实际本地测试字符串。"""
            return "正常"

        @tool
        def greeting(name: str = "同学") -> str:
            """测试未传入参数时沿用函数默认值。"""
            return "你好，" + name

        self.assertEqual(execute_tool("ping", {}, [ping])["result"], "正常")
        self.assertEqual(execute_tool("greeting", {}, [greeting])["result"], "你好，同学")

    def test_executor_rejects_unknown_duplicate_and_non_dict_arguments(self):
        for name, args, tools in (("unknown", {}, self.tools), ("multiply", [], self.tools),
                                  ("multiply", {}, self.tools * 2)):
            with self.subTest(name=name, args=args):
                self.assertEqual(execute_tool(name, args, tools)["status"], "error")
        self.assertEqual(self.invocations, [])

    def test_structured_results_and_mutating_tools_preserve_original_parameters(self):
        @tool
        def annotate(items: list[str]) -> dict:
            """修改执行副本，返回真实结构化结果。"""
            items.append("新增")
            return {"items": items, "source_file": "测试.txt"}

        args = {"items": ["原始"]}
        result = execute_tool("annotate", args, [annotate])
        self.assertEqual(args, {"items": ["原始"]})
        self.assertEqual(result["args"], args)
        self.assertEqual(result["result"]["items"], ["原始", "新增"])

    def test_closing_after_call_event_does_not_execute_tool(self):
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.response).encode())):
            events = act("3.14乘以2.56", self.thought, self.tools)
            self.assertEqual(next(events)["type"], "tool_call")
            self.assertEqual(self.invocations, [])
            events.close()
        self.assertEqual(self.invocations, [])

    def test_execution_keeps_selected_tool_when_caller_changes_registry_after_call_event(self):
        @tool
        def multiply(a: float, b: float) -> float:
            """同名替代工具不应替换已经规划好的本次调用。"""
            raise AssertionError("不能中途换工具")

        tools = self.tools.copy()
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.response).encode())):
            events = act("问题", self.thought, tools)
            next(events)
            tools[:] = [multiply]
            result = next(events)
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.invocations, [(3.14, 2.56)])

    def test_partial_response_service_errors_and_malformed_json_do_not_execute(self):
        for response in ([], {"error": "模型调用失败"}, {**self.response, "done": False},
                         {**self.response, "done_reason": "length"}, {**self.response, "message": []},
                         {**self.response, "model": None}):
            with self.subTest(response=response):
                self.assertEqual(self.run_action(response)[0]["type"], "error")
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(b"invalid-json")):
            self.assertEqual(list(act("问题", self.thought, self.tools))[0]["type"], "error")
        self.assertEqual(self.invocations, [])

    def test_network_failure_and_cloud_configuration_do_not_execute(self):
        with patch("src.agent.react_loop.urlopen", side_effect=URLError("失败")) as http:
            events = list(act("问题", self.thought, self.tools))
        self.assertEqual(http.call_count, 1)
        self.assertEqual(events[0]["type"], "error")
        config = deepcopy(load_config())
        config["llm"]["base_url"] = "http://example.com"
        with patch("src.agent.react_loop.load_config", return_value=config), patch("src.agent.react_loop.urlopen") as http:
            self.assertEqual(list(act("问题", self.thought, self.tools))[0]["type"], "error")
            http.assert_not_called()
        self.assertEqual(self.invocations, [])


if __name__ == "__main__":
    unittest.main()
