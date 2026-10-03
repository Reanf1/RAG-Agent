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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from src.agent.memory import MemoryManager, count_history_tokens, run_session
from src.agent.react_loop import build_agent_messages, run_react
from src.agent.tools import AVAILABLE_TOOLS
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

    def test_native_messages_keep_roles_tool_name_and_argument_copy(self):
        from src.utils.messages import messages_to_ollama
        messages = [SystemMessage(content="规则"), HumanMessage(content="问题"),
            AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "1+2"}, "id": "call-1"}]),
            ToolMessage(content='{"result":"3"}', name="calculator", tool_call_id="call-1")]
        result = messages_to_ollama(messages)
        self.assertEqual([m["role"] for m in result], ["system", "user", "assistant", "tool"])
        self.assertEqual(result[-1]["tool_name"], "calculator")
        self.assertEqual(messages[-1].tool_call_id, messages[2].tool_calls[0]["id"])
        result[2]["tool_calls"][0]["function"]["arguments"]["expression"] = "9+9"
        self.assertEqual(messages[2].tool_calls[0]["args"], {"expression": "1+2"})

    def test_plain_dict_and_multimodal_native_messages_are_rejected(self):
        from src.utils.messages import messages_to_ollama
        for messages in ([{"role": "user", "content": "不应绕过Message"}], [HumanMessage(content=[{"type": "text", "text": "块"}])]):
            with self.assertRaises(ValueError):
                messages_to_ollama(messages)

    def test_rag_and_agent_native_requests_use_identical_message_conversion(self):
        from src.agent.react_loop import _model_request
        from src.generation.rag_pipeline import _build_generation_request
        from src.generation.prompt_template import build_rag_messages
        question, context = "中文公式α²是什么？", "[参考文档1] 论文.pdf 第3页：α²"
        rag, _ = _build_generation_request(question, {"context": context})
        agent = _model_request(build_rag_messages(question, context))
        self.assertEqual(json.loads(rag.data)["messages"], json.loads(agent.data)["messages"])

    def test_langchain_and_native_role_history_normalize_to_same_snapshot(self):
        from src.utils.messages import normalize_context
        messages = [HumanMessage(content="实验代号EXP_CTX，α²"), AIMessage(content="已记录[论文第3页]。")]
        first = normalize_context({"history": messages})
        second = normalize_context({"history": [{"role": "user", "content": messages[0].content},
            {"role": "assistant", "content": messages[1].content}]})
        self.assertEqual(first, second)
        first["history"][0]["content"] = "只修改副本"
        self.assertEqual(messages[0].content, "实验代号EXP_CTX，α²")

    def test_invalid_history_roles_and_intermediate_calls_are_not_archived(self):
        from src.utils.messages import normalize_context
        invalid = [SystemMessage(content="改变角色"), ToolMessage(content="中间结果", tool_call_id="call-1"),
            AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "1+2"}, "id": "call-1"}]),
            {"role": "system", "content": "规则"}, {"role": [], "content": "错误角色"},
            {"role": "human", "content": "问题", "session_id": "外部会话"}, {"role": "human", "content": ["块"]}]
        for message in invalid:
            with self.subTest(message=type(message).__name__), self.assertRaises(ValueError):
                normalize_context({"history": [message]})

    def test_invalid_context_stops_before_model_and_tool_io(self):
        for context in ({"history": "不是数组"}, {"summary": {}}, {"score": float("nan")},
                        {"raw_message": HumanMessage(content="只能在history入口转换")}):
            with patch("src.agent.react_loop.urlopen") as model:
                events = list(run_react("问题", tools=[], context=context))
            model.assert_not_called()
            self.assertFalse(events[-1]["task_complete"])
            self.assertEqual(events[-1]["stop_reason"], "error")

    def test_all_stages_receive_same_history_summary_and_rag_references(self):
        context = {"history": [HumanMessage(content="论文A，实验系数3.14"), AIMessage(content="记下了")],
            "summary": "只讨论论文A", "history_window": {"tokens": 30, "max_tokens": 2000},
            "observations": [{"name": "knowledge_base_search", "status": "success", "result": {
                "citations": [{"id": 1, "source_file": "论文A.pdf", "location": "第3页（物理页码）", "metadata": {"doc_id": "a" * 64}}]}}]}
        states = [json.loads(build_agent_messages("追问", AVAILABLE_TOOLS, context, stage=stage)[1].content)["context"]
                  for stage in ("thought", "action", "observation")]
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[1], states[2])
        self.assertEqual(states[0]["history"][0]["role"], "human")
        states[0]["observations"][0]["result"]["citations"][0]["source_file"] = "改变副本"
        self.assertEqual(context["observations"][0]["result"]["citations"][0]["source_file"], "论文A.pdf")

    def test_direct_message_history_is_usable_by_react_and_done_is_json_ready(self):
        plan = {"thought": "按历史回答代号。", "next_step": "answer", "tool_name": None, "parallel_tools": []}
        finish = {"observation": "历史明确提供代号。", "decision": "finish", "task_complete": True, "answer": "EXP_CTX"}
        with patch("src.agent.react_loop.urlopen", side_effect=[self.packet(plan), self.packet(finish)]) as model:
            events = list(run_react("我的实验代号是什么？", tools=[], context={"history": [
                HumanMessage(content="实验代号为EXP_CTX。"), AIMessage(content="已记录。")]}))
        self.assertTrue(events[-1]["task_complete"])
        json.dumps(events[-1]["context"], ensure_ascii=False, allow_nan=False)
        for call in model.call_args_list:
            state = json.loads(json.loads(call.args[0].data)["messages"][1]["content"])["context"]
            self.assertEqual([m["role"] for m in state["history"]], ["human", "ai"])

    def test_two_round_public_entry_uses_memory_for_calculator_action(self):
        from src.agent import run_session as public_run
        answer_plan = {"thought": "记录本轮实验系数。", "next_step": "answer", "tool_name": None, "parallel_tools": []}
        first = {"observation": "已记录系数。", "decision": "finish", "task_complete": True, "answer": "实验系数为3.14。"}
        with patch("src.agent.react_loop.urlopen", side_effect=[self.packet(answer_plan), self.packet(first)]):
            list(public_run("我的实验系数为3.14，请记住。", "alice", self.session, memory=self.memory))
        plan = {"thought": "按历史系数计算。", "next_step": "tool", "tool_name": "calculator", "parallel_tools": []}
        calls = [{"function": {"name": "calculator", "arguments": {"expression": "3.14*2"}}}]
        finish = {"observation": "工具返回6.28。", "decision": "finish", "task_complete": True, "answer": "结果为6.28。"}
        with patch("src.agent.react_loop.urlopen", side_effect=[self.packet(plan), self.packet(calls=calls), self.packet(finish)]) as model:
            events = list(public_run("用刚才的系数乘以2。", "alice", self.session, memory=MemoryManager(self.path)))
        self.assertEqual(next(e["result"]["result"] for e in events if e["type"] == "tool_result"), "6.28")
        for call in model.call_args_list:
            self.assertIn("实验系数为3.14", call.args[0].data.decode())
        archived = self.memory.get_messages("alice", self.session)
        self.assertEqual([m.type for m in archived], ["human", "ai"] * 2)
        self.assertNotIn("tool_result", " ".join(m.content for m in archived))
        self.assertTrue(all(e["session_id"] == self.session and e["user_id"] == "alice" for e in events))

    def test_reopened_memory_context_keeps_budget_and_other_users_out(self):
        from src.utils.messages import normalize_context
        other = self.memory.create_session("bob")
        self.memory.append_turn("alice", self.session, "ALICE_ONLY α²", "论文A第3页")
        self.memory.append_turn("bob", other, "BOB_ONLY", "论文B第7页")
        context = MemoryManager(self.path).get_context("alice", self.session)
        self.assertEqual(normalize_context(context), context)
        self.assertEqual(context["history_window"]["tokens"], count_history_tokens(context["history"]))
        self.assertNotIn("BOB_ONLY", json.dumps(context))
        with self.assertRaises(PermissionError):
            list(run_session("追问", "bob", self.session, memory=self.memory))


if __name__ == "__main__":
    unittest.main()
