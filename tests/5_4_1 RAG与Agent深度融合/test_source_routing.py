"""5.4.1 RAG与Agent深度融合：TestRAGSearchRouting。"""

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
from unittest.mock import MagicMock, patch
from langchain_core.tools import tool
from src.agent.react_loop import build_agent_messages, run_react
from src.agent.tools import knowledge_base_search
from src.agent.tools import get_available_tools, web_search
from src.agent.router import route_question
from src.utils.config import load_config
from tests.helpers import isolated_agent_logs


def setUpModule():
    """保持既有 Agent 测试的模块级日志隔离。"""
    global _log_context
    _log_context = isolated_agent_logs()
    _log_context.__enter__()


def tearDownModule():
    _log_context.__exit__(None, None, None)


class TestRAGSearchRouting(unittest.TestCase):
    """验证资料来源选择与真实工具链；模型HTTP和外网响应明确隔离。"""

    def setUp(self):
        self.config = deepcopy(load_config())
        self.config["agent"]["online_search_enabled"] = True
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config["paths"]["logs"] = self.directory.name
        for module in ("src.agent.tools", "src.agent.router", "src.agent.react_loop", "src.utils.logger"):
            patcher = patch(module + ".load_config", return_value=self.config)
            patcher.start()
            self.addCleanup(patcher.stop)

    def packet(self, content=None, calls=None):
        return {"model": "qwen2.5:7b", "done": True, "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 30,
                "message": {"tool_calls": calls} if calls is not None else {"content": json.dumps(content, ensure_ascii=False)}}

    def test_latest_external_papers_route_to_enabled_search(self):
        for question in ("最新的Transformer论文有哪些？", "近期大模型研究进展", "今年有哪些人工智能论文？",
                         "What are the latest papers on AI?", "Recent advances in computer vision"):
            with self.subTest(question=question):
                plan = route_question(question, get_available_tools())
                self.assertEqual(plan["tool_name"], "web_search")
                self.assertEqual(plan["usage"], {"prompt_eval_count": 0, "eval_count": 0})

    def test_local_scope_beats_freshness_words(self):
        for question in ("知识库中最新论文用了什么数据集？", "已上传论文中提到的最新进展是什么？",
                         "本文的最近研究结果是什么？", "What are the latest results in this paper?",
                         "最新实验结果，文献ID：" + "a" * 64):
            with self.subTest(question=question):
                self.assertEqual(route_question(question, get_available_tools())["tool_name"], "knowledge_base_search")

    def test_existing_local_and_general_concept_routes_still_work(self):
        self.assertEqual(route_question("论文用了什么数据集？", get_available_tools())["tool_name"], "knowledge_base_search")
        self.assertEqual(route_question("什么是Transformer？", get_available_tools())["next_step"], "answer")
        self.assertEqual(route_question("查询已上传论文的作者和年份", get_available_tools())["tool_name"], "paper_metadata")

    def test_explicit_registered_tool_is_not_replaced_by_freshness(self):
        self.assertEqual(route_question("用knowledge_base_search检索最新论文", get_available_tools())["tool_name"], "knowledge_base_search")
        self.assertEqual(route_question("用paper_summary总结最新上传的论文", get_available_tools())["tool_name"], "paper_summary")

    def test_external_search_phrases_route_when_enabled(self):
        for question in ("上网查找Transformer论文", "联网检索ViT论文", "search the internet for AI papers", "search online for Transformer"):
            with self.subTest(question=question):
                self.assertEqual(route_question(question, get_available_tools())["tool_name"], "web_search")

    def test_disabled_search_never_routes_latest_query_to_local_rag(self):
        self.config["agent"]["online_search_enabled"] = False
        for question in ("最新Transformer论文有哪些？", "联网搜索论文", "latest papers on AI"):
            plan = route_question(question, get_available_tools())
            self.assertEqual(plan["unavailable_tool"], "web_search")
            self.assertIsNone(plan["tool_name"])
        self.assertNotIn(web_search, get_available_tools())

    def test_missing_tools_do_not_create_fake_routes(self):
        self.assertEqual(route_question("最新论文有哪些？", [knowledge_base_search])["unavailable_tool"], "web_search")
        self.assertIsNone(route_question("知识库中最新结果是什么？", [web_search]))

    def test_mixed_local_and_external_sources_need_model_planning(self):
        for question in ("查询知识库里的Transformer结果，同时联网搜索最新进展", "已上传论文的实验结果以及最新外部进展",
                         "先查询知识库，再联网查找近期论文"):
            self.assertIsNone(route_question(question, get_available_tools()))

    def test_existing_context_and_negation_are_not_overridden(self):
        self.assertIsNone(route_question("最新论文有哪些？", get_available_tools(), {"history": [{"role": "human", "content": "论文主题"}]}))
        self.assertIsNone(route_question("不要联网，说明知识库中的最新结果", get_available_tools()))

    def test_explicit_current_document_and_tool_route_despite_old_history(self):
        """复现W09/W10：不能把新指定文档或强制检索替换为旧答案。"""
        context = {"history": [{"role": "ai", "content": "旧ViT答案"}]}
        for question in ("请根据知识库文档" + "a" * 64 + "回答复测代号是什么？",
                         "必须实际调用knowledge_base_search查询ViT的位置编码"):
            with self.subTest(question=question):
                plan = route_question(question, get_available_tools(), context)
                self.assertIsNotNone(plan)
                self.assertEqual(plan["tool_name"], "knowledge_base_search")
                self.assertEqual(plan["next_step"], "tool")

    def filename_tools(self, papers):
        """可控工具结果测试预检契约，不运行真实模型或写入知识库。"""
        self.search_calls = []
        @tool
        def paper_list() -> dict:
            """返回给定真实列表样例。"""
            return {"papers": papers}
        @tool
        def knowledge_base_search(question: str, doc_id: str | None = None) -> dict:
            """记录实际目标，返回已完成任务样例。"""
            self.search_calls.append((question, doc_id))
            return {"answer": "真实测试工具结果", "status": "answered", "generation_mode": "grounded", "citations": [{"id": 1}]}
        return [paper_list, knowledge_base_search]

    def test_filename_preflight_continues_to_requested_knowledge_tool(self):
        identifier = "a" * 64
        tools = self.filename_tools([{"doc_id": identifier, "source_file": "论文.pdf"}])
        action = self.packet(calls=[{"function": {"name": "knowledge_base_search", "arguments": {
            "question": "方法是什么？", "doc_id": identifier}}}])
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(action).encode())) as http:
            events = list(run_react("请使用knowledge_base_search查询论文.pdf：方法是什么？", tools))
        self.assertTrue(events[-1]["task_complete"])
        self.assertEqual(self.search_calls, [("方法是什么？", identifier)])
        self.assertEqual([e["name"] for e in events if e["type"] == "tool_result"], ["paper_list", "knowledge_base_search"])
        self.assertEqual(http.call_count, 1)

    def test_missing_or_ambiguous_filename_does_not_search_other_documents(self):
        for papers in ([{"doc_id": "a" * 64, "source_file": "另一篇.pdf"}],
                       [{"doc_id": code * 64, "source_file": "论文.pdf"} for code in ("a", "b")]):
            tools = self.filename_tools(papers)
            with self.subTest(papers=papers), patch("src.agent.react_loop.urlopen") as http:
                events = list(run_react("请查询论文.pdf的方法", tools))
            self.assertFalse(events[-1]["task_complete"])
            self.assertEqual(events[-1]["stop_reason"], "incomplete")
            self.assertIn("唯一匹配", events[-1]["full_response"])
            self.assertEqual(self.search_calls, [])
            http.assert_not_called()

    def test_source_policy_is_present_in_all_stages(self):
        for stage in ("thought", "action", "observation"):
            system = build_agent_messages("问题", get_available_tools(), stage=stage)[0].content
            for text in ("【资料来源决策】", "已上传", "最新", "不自动联网", "网页URL"):
                self.assertIn(text, system)

    def test_disabled_search_policy_uses_actual_registry(self):
        self.config["agent"]["online_search_enabled"] = False
        messages = build_agent_messages("最新论文？", get_available_tools())
        self.assertIn("联网搜索当前不可用", messages[0].content)
        self.assertIn("task_complete=false", messages[0].content)

    def test_enabled_search_policy_reports_availability(self):
        self.assertIn("联网搜索当前可用", build_agent_messages("问题", get_available_tools())[0].content)
        self.assertIn("联网搜索当前不可用", build_agent_messages("问题", [knowledge_base_search])[0].content)

    def test_rag_action_observation_keeps_actual_sources_and_calls_retrieval_once(self):
        from langchain_core.documents import Document
        from src.utils.logger import read_rag_requests

        document = Document(page_content="Transformer在WMT14上取得28.4 BLEU。", metadata={"doc_id": "a" * 64,
            "chunk_id": "chunk-1", "source_file": "论文.pdf", "page_number": 3, "file_type": ".pdf"})
        generation = {**self.packet(), "message": {"content": "实验结果为28.4 BLEU[参考文档1]。"}}
        finish = {"observation": "已返回论文证据。", "decision": "finish", "task_complete": True, "answer": "28.4 BLEU（论文.pdf，第3页）。"}
        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.generation.rag_pipeline.urlopen", return_value=BytesIO(json.dumps(generation).encode())), \
                patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(self.packet(finish)).encode())) as model, \
                patch("httpx.Client") as network:
            retriever.return_value.search.return_value = [(document, 0.95)]
            events = list(run_react("这篇论文的实验结果？"))
        retriever.return_value.search.assert_called_once_with("这篇论文的实验结果？", doc_id=None, rerank=True)
        self.assertEqual(model.call_count, 0)  # 明确单个RAG任务保留工具答案，不二次生成引用。
        result = next(event["result"] for event in events if event["type"] == "tool_result")
        self.assertEqual(result["citations"][0]["location"], "第3页（物理页码）")
        self.assertEqual(events[-1]["context"]["observations"][0]["result"]["citations"], result["citations"])
        self.assertTrue(events[-1]["task_complete"])
        self.assertEqual(read_rag_requests()[0][0]["request_info"]["entrypoint"], "knowledge_base_search")
        network.assert_not_called()

    def test_enabled_search_runs_real_parser_without_touching_rag(self):
        response = MagicMock(status_code=200, text='<div class="result"><a class="result__a" href="https://example.org/paper">测试网页</a><a class="result__snippet">明确构造的网页摘要</a></div>')
        calls = [{"function": {"name": "web_search", "arguments": {"query": "最新Transformer论文"}}}]
        finish = {"observation": "已返回网页摘要。", "decision": "finish", "task_complete": True, "answer": "测试网页：https://example.org/paper（网页摘要）"}
        with patch("httpx.Client") as client, patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.agent.react_loop.urlopen", side_effect=[BytesIO(json.dumps(self.packet(calls=calls)).encode()),
                    BytesIO(json.dumps(self.packet(finish)).encode())]):
            client.return_value.__enter__.return_value.get.return_value = response
            events = list(run_react("最新Transformer论文有哪些？"))
        retriever.assert_not_called()
        result = next(event["result"] for event in events if event["type"] == "tool_result")
        self.assertEqual(result["results"][0]["url"], "https://example.org/paper")
        self.assertNotIn("citations", result)
        self.assertTrue(events[-1]["task_complete"])

    def test_disabled_latest_request_finishes_incomplete_without_tools(self):
        self.config["agent"]["online_search_enabled"] = False
        with patch("src.agent.react_loop.urlopen") as model, patch("httpx.Client") as network, \
                patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever:
            events = list(run_react("最新Transformer论文有哪些？"))
        model.assert_not_called()
        network.assert_not_called()
        retriever.assert_not_called()
        self.assertFalse(events[-1]["task_complete"])
        self.assertEqual(events[-1]["stop_reason"], "incomplete")

    def test_disabled_latest_request_with_history_still_skips_all_io(self):
        self.config["agent"]["online_search_enabled"] = False
        self.assertIsNone(route_question("最近的代号是什么？", get_available_tools(), {"history": [{"role": "human", "content": "代号为EXP1"}]}))
        with patch("src.agent.react_loop.urlopen") as model, patch("httpx.Client") as network:
            events = list(run_react("最新Transformer论文有哪些？", context={"history": [{"role": "human", "content": "研究主题为Transformer"}]}))
        model.assert_not_called()
        network.assert_not_called()
        self.assertEqual(events[-1]["stop_reason"], "incomplete")
        self.assertFalse(any(event["type"] == "tool_call" for event in events))

    def test_search_failure_cannot_recover_via_local_old_papers(self):
        import httpx
        calls = [{"function": {"name": "web_search", "arguments": {"query": "最新Transformer论文"}}}]
        finish = {"observation": "联网搜索失败，没有最新外部证据。", "decision": "finish", "task_complete": False, "answer": "联网搜索失败，请稍后重试。"}
        with patch("httpx.Client") as client, patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.agent.react_loop.urlopen", side_effect=[BytesIO(json.dumps(self.packet(calls=calls)).encode()),
                    BytesIO(json.dumps(self.packet(finish)).encode())]) as model:
            client.return_value.__enter__.return_value.get.side_effect = httpx.ConnectError("测试网络断开")
            events = list(run_react("最新Transformer论文有哪些？"))
        retriever.assert_not_called()
        self.assertFalse(events[-1]["task_complete"])
        self.assertEqual(events[-1]["context"]["recovery"]["available_alternatives"], [])
        self.assertEqual([t["function"]["name"] for t in json.loads(model.call_args_list[0].args[0].data)["tools"]], ["web_search"])

    def test_grounded_rag_without_valid_citations_is_not_completed(self):
        from langchain_core.documents import Document
        from src.utils.logger import read_rag_requests
        document = Document(page_content="模型取得28.4 BLEU。", metadata={"doc_id": "a" * 64, "chunk_id": "chunk-1",
            "source_file": "论文.pdf", "page_number": 3, "file_type": ".pdf"})
        generation = {**self.packet(), "message": {"content": "结果为28.4 BLEU。"}}
        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.generation.rag_pipeline.urlopen", return_value=BytesIO(json.dumps(generation).encode())):
            retriever.return_value.search.return_value = [(document, 0.95)]
            result = knowledge_base_search.invoke({"question": "论文结果？"})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["citations"], [])
        self.assertEqual(read_rag_requests()[0][0]["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
