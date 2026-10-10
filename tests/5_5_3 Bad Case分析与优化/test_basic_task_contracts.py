"""课程基本任务回归：验证原问题、文档来源、数字证据和失败记录的一致性。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from copy import deepcopy
from io import BytesIO
import json
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.tools import tool
from src.agent import react_loop, router, tools
from src.generation import rag_pipeline, streaming
from src.retrieval import hybrid_retriever
from src.utils import logger


DOC_ID = "a" * 64


def packet(content="", calls=None):
    """仅替换模型接口，保留实际工具执行和引用、日志处理。"""
    message = {"content": content}
    if calls is not None:
        message["tool_calls"] = calls
    return {"model": "test", "done": True, "done_reason": "stop", "message": message,
            "prompt_eval_count": 101, "eval_count": 23}


class TestBasicTaskContracts(unittest.TestCase):

    def test_keyword_route_does_not_depend_on_filename_length(self):
        for filename in ("甲叶脉.txt", "澄禾实验.txt", "课程实践中的作物病害识别实验说明.md"):
            with self.subTest(filename=filename):
                plan = router.route_question(f"提取{filename}的关键词", tools.AVAILABLE_TOOLS)
                self.assertEqual(plan["tool_name"], "keyword_extract")
        plan = router.route_question("Extract experiment-notes.txt keywords", tools.AVAILABLE_TOOLS)
        self.assertEqual(plan["tool_name"], "keyword_extract")

    def test_document_keywords_read_document_instead_of_model_written_text(self):
        seen = []

        @tool
        def keyword_extract(text: str | None = None, doc_id: str | None = None) -> dict:
            """记录实际输入；文档任务应只携带文档ID。"""
            seen.append((text, doc_id))
            return {"doc_id": doc_id, "keywords": ["叶脉"]}

        listing = {"name": "paper_list", "status": "success", "result": {
            "papers": [{"doc_id": DOC_ID, "source_file": "澄禾实验.txt"}]}}
        thought = {"next_step": "tool", "tool_name": "keyword_extract", "tool_calls": [{"function": {
            "name": "keyword_extract", "arguments": {"text": "模型自己编写的内容"}}}]}
        events = list(react_loop.act("提取澄禾实验.txt的关键词", thought, [keyword_extract],
                                    {"observations": [listing]}))
        self.assertEqual(events[-1]["status"], "success")
        self.assertEqual(seen, [(None, DOC_ID)])


    def test_subquestion_result_still_needs_observation_of_whole_task(self):
        question = "本地论文使用哪个数据集，准确率是多少？"
        evidence = {"name": "knowledge_base_search", "status": "success", "args": {"question": "使用哪个数据集？"},
                    "result": {"status": "answered", "answer": "使用田畴数据集。[参考文档1]",
                               "citations": [{"id": 1}], "generation_mode": "grounded", "done_reason": "stop"}}
        pending = {"observation": "还需查询准确率。", "decision": "continue", "task_complete": False, "answer": ""}
        with patch.object(react_loop, "urlopen", return_value=BytesIO(json.dumps(packet(json.dumps(pending))).encode())) as http:
            result = react_loop.observe(question, [tools.knowledge_base_search], {"observations": [evidence]})
        self.assertEqual(http.call_count, 1)
        self.assertFalse(result["task_complete"])


    def test_rag_failure_keeps_same_partial_answer_and_tokens_in_log(self):
        document = Document(page_content="Aurora uses FIELD-73 and reaches 94.2%.", metadata={
            "doc_id": DOC_ID, "chunk_id": "probe", "source_file": "核验.txt", "line_start": 1, "line_end": 1})

        class Retriever:
            def __init__(self):
                self.vector_store = self

            def list_chunks(self, **kwargs):
                return [deepcopy(document)]

            def search(self, *args, **kwargs):
                return [(deepcopy(document), .99)]

        raw = '原文依据："Aurora uses FIELD-73 and reaches 94.2%."[参考文档1]'
        invalid = packet(raw)
        invalid.pop("model")
        for stream in (False, True):
            with self.subTest(stream=stream):
                logs, tokens = [], []
                packets = [{"message": {"content": raw}, "done": False}, {**invalid, "message": {"content": ""}}]
                response = BytesIO(("\n".join(json.dumps(value) for value in packets) if stream else json.dumps(invalid)).encode())
                module = streaming if stream else rag_pipeline
                with patch.object(hybrid_retriever, "HybridRetriever", Retriever), \
                        patch.object(logger, "_append_record", lambda prefix, row: logs.append(deepcopy(row))), \
                        patch.object(module, "urlopen", return_value=response):
                    result = tools.execute_tool("knowledge_base_search", {"question": "数据集和准确率是什么？"},
                                                tools.AVAILABLE_TOOLS, on_token=tokens.append if stream else None)
                self.assertEqual(result["status"], "error")
                self.assertEqual(logs[-1]["status"], "error")
                self.assertEqual(sum(result["usage"].values()), 124)
                self.assertEqual(logs[-1]["tokens"]["total"], 124)
                self.assertEqual(logs[-1]["raw_answer"], raw)
                self.assertIn("94.2%", logs[-1]["answer"])


if __name__ == "__main__":
    unittest.main()
