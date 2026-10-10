"""课程基本任务回归：验证原问题、文档来源、数字证据和失败记录的一致性。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from copy import deepcopy
from io import BytesIO
import json
from time import perf_counter
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.tools import tool
from src.agent import react_loop, router, tools
from src.generation import rag_pipeline, streaming
from src.retrieval import hybrid_retriever, reranker
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
    def test_document_question_about_comparison_is_not_a_compare_command(self):
        question = "Windows复测_说明.md中的Markdown标记是什么？对比两篇论文时需要保留哪三个方面？请给出来源。"
        plan = router.route_question(question, tools.AVAILABLE_TOOLS)
        self.assertEqual(plan["tool_name"], "knowledge_base_search")
        for question in ("对比两篇论文", "请比较甲.txt和乙.txt两篇论文的方法和结果", "Compare these two papers"):
            with self.subTest(question=question):
                self.assertEqual(router.route_question(question, tools.AVAILABLE_TOOLS)["tool_name"], "paper_compare")

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

    def test_completed_single_tool_enters_answer_stage_with_current_question(self):
        # 工具已返回所需结果时只能收尾；历史任务不能让同一请求继续规划。
        for question, name, result in [
            ("提取澄禾实验.txt的关键词", "keyword_extract", {"keywords": ["LeafGate"], "doc_id": DOC_ID}),
            ("查询澄禾实验.txt的论文元信息，只需标题、作者和年份。", "paper_metadata",
             {"title": "澄禾实验", "authors": ["李禾"], "year": 2025, "doc_id": DOC_ID}),
        ]:
            with self.subTest(name=name):
                context = {"history": [{"role": "human", "content": "比较两篇论文的数据集和准确率。"}],
                    "observations": [{"name": name, "status": "success", "args": {"doc_id": DOC_ID}, "result": result}]}
                answer = {"observation": "所需信息已齐全。", "decision": "finish", "task_complete": True, "answer": "有依据的结果"}
                with patch.object(react_loop, "urlopen", return_value=BytesIO(json.dumps(packet(json.dumps(answer))).encode())) as http:
                    react_loop.observe(question, tools.AVAILABLE_TOOLS, context)
                request = json.loads(http.call_args.args[0].data)
                self.assertEqual(request["format"]["properties"]["decision"], {"const": "finish"})
                self.assertIn(question, request["messages"][-1]["content"])
                self.assertNotIn("history", request["messages"][-1]["content"])

    def test_keywords_keep_verified_terms_when_model_adds_a_synonym(self):
        # 模型偶尔扩展一个词，不应丢弃其余已能定位的原文关键词。
        source = "通过图像分类识别水稻叶片病害。方法：LeafGate分类器。"
        selection = {"keywords": ["水稻叶片病害", "LeafGate", "深度学习"]}
        with patch.object(tools, "_tool_model_response", return_value=(packet(), selection)):
            result = tools.keyword_extract.invoke({"text": source})
        self.assertEqual(result["keywords"], ["水稻叶片病害", "LeafGate"])
        self.assertEqual([row["keyword"] for row in result["evidence"]], result["keywords"])

    def test_single_rag_action_preserves_all_parts_of_original_question(self):
        @tool
        def knowledge_base_search(question: str, doc_id: str | None = None) -> dict:
            """回传实际收到的问题。"""
            return {"question": question}

        question = f"本地论文 {DOC_ID} 使用哪个数据集，准确率是多少？"
        thought = {"next_step": "tool", "tool_name": "knowledge_base_search", "route": "rule",
                   "tool_calls": [{"function": {"name": "knowledge_base_search",
                                  "arguments": {"question": "使用哪个数据集？", "doc_id": DOC_ID}}}]}
        events = list(react_loop.act(question, thought, [knowledge_base_search]))
        self.assertEqual(events[-1]["result"]["question"], question)

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

    def test_comparison_accepts_normal_metric_statements_and_adjacent_context(self):
        ids = [DOC_ID, "b" * 64]
        documents = [Document(page_content=f"方法：{model}分类器。数据集：田畴-73。实验结果：{model}测试准确率为{value}%。",
            metadata={"doc_id": identifier, "chunk_id": label, "source_file": label + ".txt", "line_start": 1, "line_end": 3})
            for label, identifier, model, value in [("甲", ids[0], "LeafGate", 91.6), ("乙", ids[1], "DenseCrop", 88.4)]]
        context = rag_pipeline.prepare_rag_context("对比方法、数据集、实验结果", [(doc, .99) for doc in documents])
        base = {"papers": [{"references": [ref for ref in context["references"] if ref["metadata"]["doc_id"] == identifier],
                            "truncated": False} for identifier in ids], "missing_dimensions": [], "low_relevance_dimensions": []}

        def select(messages, schema, name):
            return packet(), {key: {"text": "准确率91.6%" if key.endswith("_a") else "准确率88.4%",
                              "reference_id": spec["properties"]["reference_id"]["enum"][0]}
                              for key, spec in schema["properties"].items()}

        with patch.object(tools, "_tool_model_response", select):
            result = tools._finish_paper_compare(context, base, [Path("甲.txt"), Path("乙.txt")],
                                                *ids, perf_counter(), confirmed=True)
        self.assertEqual(result["status"], "answered")
        self.assertIn("91.6", result["answer"])
        self.assertIn("88.4", result["answer"])
        self.assertEqual({ref["metadata"]["doc_id"] for ref in result["citations"]}, set(ids))

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

    def test_reranking_preserves_retrieval_notice_when_excerpt_changes(self):
        notice = "持久化向量读取失败（Label not found）；原索引完整性仍待核验。"
        document = Document(page_content="实际正文", metadata={"retrieval_warning": notice, "source_file": "核验.txt"})

        class Model:
            tokenizer = None

            def predict(self, pairs, **kwargs):
                return [.8] * len(pairs)

        with patch.object(reranker, "get_reranker", return_value=Model()), \
                patch.object(reranker.Reranker, "_passages", return_value=["前部", "尾部"]):
            ranked = reranker.Reranker().rerank("问题", [(document, .9)])
        context = rag_pipeline.prepare_rag_context("问题", ranked)
        answer = rag_pipeline.resolve_citations("依据原文。[参考文档1]", context)
        self.assertIn(notice, answer["warnings"])
        self.assertTrue(any("摘录" in warning for warning in answer["warnings"]))
        with patch.object(reranker, "get_reranker", return_value=Model()), \
                patch.object(reranker.Reranker, "_passages", return_value=["实际正文"]):
            short = reranker.Reranker().rerank("问题", ranked)[0][0]
        self.assertEqual(short.metadata["retrieval_warning"], notice)
        self.assertNotIn("rerank_excerpt", short.metadata)
        self.assertEqual(document.metadata["retrieval_warning"], notice)


if __name__ == "__main__":
    unittest.main()
