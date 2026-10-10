"""5.4.1 RAG与Agent深度融合：TestRAGSearchRouting。"""

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
from langchain_core.tools import tool
from src.agent.react_loop import run_react
from src.agent.tools import knowledge_base_search
from src.agent.tools import get_available_tools
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


    def test_named_local_file_content_routes_to_search_with_or_without_history(self):
        """复现Windows代号问题：TXT文件名本身就是本地资料目标。"""
        for context in (None, {"history": [{"role": "ai", "content": "另一份文档的旧答案"}]}):
            for filename in ("Windows复测_说明.txt", "记录.md", "研究.docx", "实验.pdf"):
                with self.subTest(context=context, filename=filename):
                    plan = route_question(f"请根据{filename}回答：复测代号是什么？请注明来源。", get_available_tools(), context)
                    self.assertIsNotNone(plan)
                    self.assertEqual(plan["tool_name"], "knowledge_base_search")
        self.assertEqual(route_question("从记录.md提取关键词", get_available_tools())["tool_name"], "keyword_extract")
        self.assertIsNone(route_question("先查询记录.md，再提取结果的关键词", get_available_tools()))


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


if __name__ == "__main__":
    unittest.main()
