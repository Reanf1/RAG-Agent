"""5.3.2 工具集开发：TestComparisonAndKeywords。"""

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
from urllib.error import URLError
from src.agent.react_loop import observe
from src.agent.tools import AVAILABLE_TOOLS, execute_tool, paper_compare, keyword_extract
from src.utils.config import load_config
from tests.helpers import isolated_agent_logs


def setUpModule():
    """保持既有 Agent 测试的模块级日志隔离。"""
    global _log_context
    _log_context = isolated_agent_logs()
    _log_context.__enter__()


def tearDownModule():
    _log_context.__exit__(None, None, None)


class TestComparisonAndKeywords(unittest.TestCase):
    """真实上传、上下文和引用解析，模型HTTP/检索结果隔离；不将模拟结果当质量数据。"""

    def setUp(self):
        from src.data_loader import batch_import, create_import_tasks

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = deepcopy(load_config())
        self.config["paths"]["raw_documents"] = self.directory.name
        self.addCleanup(patch.stopall)
        for module in ("src.agent.tools", "src.agent.react_loop", "src.generation.rag_pipeline"):
            patch(module + ".load_config", return_value=self.config).start()
        tasks = create_import_tasks([("同名.md", b"Transformer method\nDataset WMT\nBLEU 28.4"),
                                     ("同名.md", b"ViT method\nDataset ImageNet\nAccuracy 88.55%")])
        list(batch_import(tasks, self.directory.name))
        from src.chunking import split_documents
        self.documents = [split_documents(task["documents"])[0] for task in tasks]
        self.ids = [document.metadata["doc_id"] for document in self.documents]
        self.response = {"model": "qwen2.5:7b", "done": True, "done_reason": "stop",
                         "prompt_eval_count": 200, "eval_count": 80,
                         "message": {"content": json.dumps({key: 1 for key in ("method", "datasets", "results")})}}

    def comparison_packet(self, request, timeout):
        """按本篇实际Schema选择首个合法编号，仅用于协议/来源测试，不冒充模型质量。"""
        properties = json.loads(request.data)["format"]["properties"]
        choice = {key: value["enum"][0] for key, value in properties.items()}
        return BytesIO(json.dumps({**self.response, "message": {"content": json.dumps(choice)}}).encode())

    def compare(self, scores=(0.9, 0.9), response=None):
        def search(query, *, k, doc_id, rerank):
            index = self.ids.index(doc_id)
            return [] if scores[index] is None else [(self.documents[index], scores[index])]

        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.generation.rag_pipeline.urlopen", side_effect=self.comparison_packet if response is None else
                      lambda request, timeout: BytesIO(json.dumps(response).encode())) as http:
            retriever.return_value.search.side_effect = search
            result = paper_compare.invoke(dict(zip(("paper_a_id", "paper_b_id"), self.ids)))
        return result, retriever, http

    def test_low_comparison_confirmation_reuses_reviewed_candidates(self):
        """确认只复用本会话、同索引版本的证据；不二次检索更换原文。"""
        from src.agent.tools import get_available_tools
        pending, args = {}, dict(zip(("paper_a_id", "paper_b_id"), self.ids))
        with patch("src.generation.cache.cache_scope", return_value="scope-v1"), \
                patch("src.retrieval.vector_store.VectorStore"), \
                patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.generation.rag_pipeline.urlopen", side_effect=self.comparison_packet) as http:
            retriever.return_value.search.side_effect = [[(self.documents[i], 0.01)] for i in ([0] * 6 + [1] * 6)]
            registry = get_available_tools(session_id="session", pending=pending, request_question="对比两篇论文")
            result = next(t for t in registry if t.name == "paper_compare").invoke(args)
            self.assertEqual(result["status"], "needs_confirmation")
            self.assertEqual({ref["metadata"]["doc_id"] for ref in result["references"]}, set(self.ids))
            approval = pending[result["confirmation_id"]]
            http.assert_not_called()
            retriever.reset_mock()
            registry = get_available_tools(session_id="session", confirmation=approval)
            confirmed = next(t for t in registry if t.name == "paper_compare").invoke(args)
            self.assertTrue(confirmed["confirmed"])
            self.assertEqual(confirmed["status"], "answered")
            retriever.assert_not_called()
            registry = get_available_tools(session_id="other", confirmation=approval)
            with self.assertRaisesRegex(ValueError, "会话"):
                next(t for t in registry if t.name == "paper_compare").invoke(args)
            with patch("src.generation.cache.cache_scope", return_value="scope-v2"):
                registry = get_available_tools(session_id="session", confirmation=approval)
                with self.assertRaisesRegex(ValueError, "失效"):
                    next(t for t in registry if t.name == "paper_compare").invoke(args)

    def keywords(self, terms, **args):
        response = {**self.response, "message": {"content": json.dumps({"keywords": terms})}}
        with patch("src.generation.rag_pipeline.urlopen", return_value=BytesIO(json.dumps(response).encode())) as http:
            result = keyword_extract.invoke(args or {"text": "Transformer与中文人工智能：ImageNet模型准确率是多少？"})
        return result, http

    def test_new_tools_have_real_registered_parameter_schemas(self):
        self.assertEqual(set(paper_compare.args), {"paper_a_id", "paper_b_id"})
        self.assertEqual(set(keyword_extract.args), {"text", "doc_id"})
        self.assertIn(paper_compare, AVAILABLE_TOOLS)
        self.assertIn(keyword_extract, AVAILABLE_TOOLS)

    def test_comparison_filters_each_paper_and_covers_three_dimensions(self):
        result, retriever, http = self.compare()
        calls = retriever.return_value.search.call_args_list
        self.assertEqual(len(calls), 12)
        self.assertEqual([call.kwargs["doc_id"] for call in calls], [self.ids[0]] * 6 + [self.ids[1]] * 6)
        self.assertTrue(all(call.kwargs["k"] == 2 and call.kwargs["rerank"] for call in calls))
        self.assertEqual(result["status"], "answered")
        self.assertEqual(len(result["citations"]), 2)
        self.assertEqual({r["metadata"]["doc_id"] for r in result["citations"]}, set(self.ids))
        self.assertEqual(result["papers"][0]["source_file"], "同名.md")
        payloads = [json.loads(call.args[0].data) for call in http.call_args_list]
        self.assertEqual(payloads[0]["format"]["properties"]["method"]["enum"], [1, None])
        self.assertEqual(payloads[1]["format"]["properties"]["method"]["enum"], [2, None])
        self.assertIn("不可直接比较", result["answer"])
        self.assertEqual(result["usage"]["eval_count"], 160)

    def test_comparison_deduplicates_chunks_without_mutating_source(self):
        before = deepcopy(self.documents)
        result, _, _ = self.compare()
        self.assertEqual([len(p["references"]) for p in result["papers"]], [1, 1])
        self.assertEqual(self.documents, before)
        self.assertEqual([r["source_file"] for r in result["citations"]], ["同名.md", "同名.md"])

    def test_dimension_budget_keeps_lower_scored_method_evidence(self):
        """不同查询的分数不能混排挤掉已召回的主方法；总预算仍不放宽。"""
        self.config["generation"]["max_context_chars"] = 1400
        candidates = []
        for index, text in enumerate(("Method: patch Transformer. ", "Dataset: ImageNet. ", "Accuracy: 88.55%. ")):
            document = deepcopy(self.documents[0])
            document.page_content = text + "Evidence details. " * 20
            document.metadata["chunk_id"] = f"dimension-{index}"
            candidates.append(document)
        before = deepcopy(candidates)
        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.generation.rag_pipeline.urlopen", side_effect=self.comparison_packet) as http:
            retriever.return_value.search.side_effect = [
                [(candidates[index], score)] for index, score in ((0, .6), (0, .6), (1, .9), (1, .9), (2, .99), (2, .99))
            ] + [[(self.documents[1], .9)]] * 6
            result = paper_compare.invoke(dict(zip(("paper_a_id", "paper_b_id"), self.ids)))
        references = result["papers"][0]["references"]
        self.assertEqual({ref["metadata"]["chunk_id"] for ref in references}, {f"dimension-{i}" for i in range(3)})
        payload = json.loads(json.loads(http.call_args_list[0].args[0].data)["messages"][1]["content"])
        self.assertTrue(any("Method: patch Transformer" in ref["text"] for ref in payload["references"]))
        self.assertEqual(candidates, before)
        self.assertTrue(result["papers"][0]["truncated"])

    def test_comparison_preserves_vector_recovery_notice_in_answer_and_confirmation(self):
        """重建对比Context和低相关确认不能丢掉临时向量恢复限制。"""
        notice = "持久化向量读取失败（Label not found）；本次临时计算，原索引完整性仍待核验。"
        self.documents[0].metadata["retrieval_warning"] = notice
        before = deepcopy(self.documents)
        for scores, status in (((.9, .9), "answered"), ((.01, .9), "needs_confirmation")):
            with self.subTest(status=status):
                result, _, http = self.compare(scores)
                self.assertEqual(result["status"], status)
                self.assertEqual(result.get("warnings", []).count(notice), 1)
                refs = result.get("references", result["citations"])
                self.assertTrue(any(ref["metadata"].get("retrieval_warning") == notice for ref in refs))
                if status == "needs_confirmation":
                    http.assert_not_called()
        self.assertEqual(self.documents, before)

    def test_qualitative_excerpt_explicitly_reports_missing_experiment_numbers(self):
        """W04节选只有定性结果时，不把年份或页码当成实验指标数值。"""
        self.documents[1].page_content = "DETR method. COCO dataset. Comparable to Faster R-CNN in 2020."
        result, _, _ = self.compare()
        self.assertIn("论文B：实验结果数值", result["missing_dimensions"])
        self.assertIn("未提供可核验的实验指标数值", result["answer"])
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_truncated_result_does_not_display_isolated_model_scores(self):
        """上下文截断后不能只列半行数值，丢失模型、数据集和对应列。"""
        self.config["generation"]["max_context_chars"] = 500
        self.documents[0].page_content = "99.45 99.68 ImageNet accuracy " * 100
        result, _, _ = self.compare()
        row = next(row for row in result["comparison"] if row["dimension"] == "实验结果")
        self.assertTrue(row["a"]["truncated"])
        result_line = next(line for line in result["answer"].splitlines() if line.startswith("| 实验结果 |"))
        self.assertNotIn("99.45", result_line)
        self.assertIn("不能完整列出数值", result_line)

    def test_method_evidence_keeps_adjacent_actual_architecture_text(self):
        """孤立背景块补入同原文相邻块，避免本篇方法只剩背景概述。"""
        from langchain_core.documents import Document
        current = self.documents[0]
        current.page_content = "Abstract: Transformers were mostly used in NLP. "
        current.metadata.update(start_index=0, end_index=len(current.page_content))
        following_text = "Method: Split images into patches and use a Transformer encoder. BLEU 28.4."
        following = Document(page_content=following_text,
            metadata={**current.metadata, "chunk_id": "next-method", "start_index": len(current.page_content),
                      "end_index": len(current.page_content) + len(following_text)})
        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.generation.rag_pipeline.urlopen", side_effect=self.comparison_packet):
            retriever.return_value.search.side_effect = lambda query, **kw: [(self.documents[self.ids.index(kw['doc_id'])], .9)]
            retriever.return_value.vector_store.list_chunks.side_effect = [[current, following], [self.documents[1]]]
            result = paper_compare.invoke(dict(zip(("paper_a_id", "paper_b_id"), self.ids)))
        row = next(row for row in result["comparison"] if row["dimension"] == "方法")
        self.assertIn("Split images into patches", row["a"]["text"])

    def test_comparison_low_relevance_cannot_generate_or_auto_confirm(self):
        result, _, http = self.compare((0.9, 0.01))
        self.assertEqual(result["status"], "needs_confirmation")
        self.assertEqual(len(result["low_relevance_dimensions"]), 3)
        self.assertEqual(result["usage"]["eval_count"], 0)
        http.assert_not_called()

    def test_comparison_unindexed_paper_has_no_model_fallback(self):
        result, _, http = self.compare((0.9, None))
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("论文B：数据集", result["missing_dimensions"])
        self.assertEqual(result["citations"], [])
        http.assert_not_called()

    def test_comparison_dimension_low_score_is_reported_with_paper_level_threshold(self):
        scores = [0.9] * 6 + [0.01, 0.01, 0.9, 0.9, 0.9, 0.9]
        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.generation.rag_pipeline.urlopen", side_effect=self.comparison_packet) as http:
            retriever.return_value.search.side_effect = [[(self.documents[i // 6], score)] for i, score in enumerate(scores)]
            result = paper_compare.invoke(dict(zip(("paper_a_id", "paper_b_id"), self.ids)))
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["low_relevance_dimensions"], ["论文B：方法"])
        self.assertEqual(result["low_relevance_papers"], [])
        self.assertTrue(any("相关性低" in warning for warning in result["warnings"]))
        self.assertEqual(http.call_count, 2)

    def test_comparison_missing_paper_takes_priority_over_other_paper_low_score(self):
        result, _, http = self.compare((None, 0.01))
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["low_relevance_papers"], ["B"])
        http.assert_not_called()

    def test_comparison_same_missing_or_invalid_id_fails_before_retrieval(self):
        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever:
            for other in (self.ids[0], "../secret", "0" * 64):
                event = execute_tool("paper_compare", {"paper_a_id": self.ids[0], "paper_b_id": other}, AVAILABLE_TOOLS)
                self.assertEqual(event["status"], "error")
            retriever.assert_not_called()

    def test_comparison_balances_budget_and_reports_truncated_evidence(self):
        self.config["generation"]["max_context_chars"] = 500
        self.documents[0].page_content = "Transformer method. " * 300
        self.documents[1].page_content = "ViT method. " * 300
        result, _, http = self.compare()
        self.assertTrue(all(p["truncated"] for p in result["papers"]))
        self.assertEqual({r["metadata"]["doc_id"] for r in result["citations"]}, set(self.ids))
        self.assertTrue(all(r["truncated"] for r in result["citations"]))
        context = json.loads(json.loads(http.call_args.args[0].data)["messages"][1]["content"])
        self.assertLess(sum(len(ref["text"]) for ref in context["references"]), 500)
        self.assertTrue(result["warnings"])

    def test_comparison_missing_one_paper_citation_is_warned(self):
        choice = {key: None for key in ("method", "datasets", "results")}
        result, _, _ = self.compare(response={**self.response, "message": {"content": json.dumps(choice)}})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertTrue(any("未同时引用两篇" in warning for warning in result["warnings"]))

    def test_comparison_reads_and_scores_opening_chunks_with_real_reranker_contract(self):
        from src.chunking import split_documents

        opening = split_documents([self.documents[0]], strategy="fixed", chunk_size=20, chunk_overlap=0)[0]
        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.retrieval.reranker.get_reranker") as model, \
                patch("src.generation.rag_pipeline.urlopen", side_effect=self.comparison_packet):
            retriever.return_value.search.side_effect = [[(self.documents[i], 0.9)] for i in ([0] * 6 + [1] * 6)]
            retriever.return_value.vector_store.list_chunks.side_effect = [[opening], []]
            model.return_value.predict.return_value = [0.8]
            result = paper_compare.invoke(dict(zip(("paper_a_id", "paper_b_id"), self.ids)))
        self.assertEqual(result["status"], "answered")
        self.assertEqual(model.return_value.predict.call_args.args[0][0][1], opening.page_content)
        self.assertIn(opening.metadata["chunk_id"], [ref["metadata"]["chunk_id"] for ref in result["papers"][0]["references"]])

    def test_comparison_rejects_wrong_paper_ids_missing_fields_and_partial_model_choice(self):
        choice = json.loads(self.response["message"]["content"])
        for response in ({**self.response, "done_reason": "length"},
                         {**self.response, "message": {"content": json.dumps({**choice, "method": 2})}},
                         {**self.response, "message": {"content": json.dumps({**choice, "method": True})}},
                         {**self.response, "message": {"content": "{}"}}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                self.compare(response=response)

    def test_comparison_network_failure_is_a_real_error_without_retry(self):
        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.generation.rag_pipeline.urlopen", side_effect=URLError("断开")) as http:
            retriever.return_value.search.side_effect = [[(self.documents[i], 0.9)] for i in ([0] * 6 + [1] * 6)]
            event = execute_tool("paper_compare", dict(zip(("paper_a_id", "paper_b_id"), self.ids)), AVAILABLE_TOOLS)
        self.assertEqual(event["status"], "error")
        self.assertIsNone(event["result"])
        self.assertEqual(http.call_count, 1)

    def test_insufficient_comparison_cannot_be_marked_complete(self):
        response = {**self.response, "message": {"content": json.dumps({"observation": "完成", "decision": "finish",
                    "task_complete": True, "answer": "假完成"})}}
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(response).encode())), self.assertRaises(RuntimeError):
            observe("对比", AVAILABLE_TOOLS, {"observations": [{"status": "success", "result": {"status": "insufficient_evidence"}}]})

    def test_observation_schema_prevents_empty_finish_seen_in_real_keyword_call(self):
        response = {**self.response, "message": {"content": json.dumps({"observation": "已有关键词。", "decision": "finish",
                    "task_complete": True, "answer": "Transformer、机器翻译。"})}}
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(response).encode())) as http:
            observe("关键词", AVAILABLE_TOOLS)
        schema = json.loads(http.call_args.args[0].data)["format"]
        finish, ongoing = schema["anyOf"]
        for branch in (finish, ongoing):
            self.assertEqual(set(branch["required"]), {"observation", "decision", "task_complete", "answer"})
            self.assertEqual(set(branch["properties"]), set(branch["required"]))
        self.assertEqual(finish["properties"]["answer"]["minLength"], 1)
        self.assertEqual(ongoing["properties"]["answer"]["maxLength"], 0)
        self.assertFalse(ongoing["properties"]["task_complete"]["const"])

    def test_keyword_question_preserves_bilingual_terms_and_evidence(self):
        result, http = self.keywords(["transformer", "人工智能", "ImageNet"])
        self.assertEqual(result["keywords"], ["Transformer", "人工智能", "ImageNet"])
        self.assertEqual(result["evidence"][0]["locations"][0]["location"], "问题第1行")
        self.assertIsNone(result["doc_id"])
        self.assertEqual(json.loads(http.call_args.args[0].data)["format"]["properties"]["keywords"]["maxItems"], 5)

    def test_keyword_document_uses_uploaded_original_and_locations(self):
        result, _ = self.keywords(["Transformer", "WMT"], doc_id=self.ids[0])
        self.assertEqual(result["source_file"], "同名.md")
        self.assertEqual(result["evidence"][1]["locations"][0]["location"], "行2")
        self.assertEqual(result["doc_id"], self.ids[0])

    def test_keyword_rejects_ambiguous_empty_and_invalid_input_before_model(self):
        with patch("src.generation.rag_pipeline.urlopen") as http:
            for args in ({}, {"text": " "}, {"text": "AI", "doc_id": self.ids[0]}, {"doc_id": "../secret"}):
                self.assertEqual(execute_tool("keyword_extract", args, AVAILABLE_TOOLS)["status"], "error")
            http.assert_not_called()

    def test_keyword_deduplication_empty_result_and_unknown_usage(self):
        result, _ = self.keywords(["Transformer", "transformer", "人工智能"])
        self.assertEqual(result["keywords"], ["Transformer", "人工智能"])
        result, _ = self.keywords([], text="你好")
        self.assertEqual(result["keywords"], [])
        self.response.pop("prompt_eval_count")
        result, _ = self.keywords(["Transformer"])
        self.assertIsNone(result["usage"]["prompt_eval_count"])

    def test_keyword_cross_line_phrase_has_real_line_evidence(self):
        result, _ = self.keywords(["neural network"], text="neural\nnetwork研究")
        self.assertEqual(result["keywords"], ["neural network"])
        self.assertEqual([r["location"] for r in result["evidence"][0]["locations"]], ["问题第1行", "问题第2行"])

    def test_keyword_truncation_does_not_accept_terms_outside_input(self):
        self.config["generation"]["max_context_chars"] = 11
        result, _ = self.keywords(["Transformer"], text="Transformer ImageNet")
        self.assertTrue(result["input_truncated"])
        with self.assertRaises(ValueError):
            self.keywords(["ImageNet"], text="Transformer ImageNet")

    def test_keyword_invalid_expanded_or_embedded_terms_are_rejected(self):
        for terms in (["不存在"], [None], [""], "AI", ["Transformer"] * 6):
            with self.subTest(terms=terms), self.assertRaises(ValueError):
                self.keywords(terms)
        for term in ("AI", " AI ", "ＡＩ"):
            with self.subTest(term=term), self.assertRaises(ValueError):
                self.keywords([term], text="training")

    def test_keyword_unfinished_malformed_and_network_response_fails(self):
        for response in ({**self.response, "done_reason": "length"}, {"error": "模型错误"},
                         {**self.response, "message": {"content": "not JSON"}}):
            with patch("src.generation.rag_pipeline.urlopen", return_value=BytesIO(json.dumps(response).encode())), self.assertRaises(ValueError):
                keyword_extract.invoke({"text": "Transformer"})
        with patch("src.generation.rag_pipeline.urlopen", side_effect=URLError("断开")) as http:
            event = execute_tool("keyword_extract", {"text": "Transformer"}, AVAILABLE_TOOLS)
        self.assertEqual(event["status"], "error")
        self.assertEqual(http.call_count, 1)


if __name__ == "__main__":
    unittest.main()
