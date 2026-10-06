"""5.3.2 工具集开发：TestResearchTools。"""

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
from src.agent.react_loop import observe, run_react
from src.agent.tools import AVAILABLE_TOOLS, execute_tool, knowledge_base_search, paper_metadata
from src.utils.config import load_config
from tests.helpers import isolated_agent_logs


def setUpModule():
    """保持既有 Agent 测试的模块级日志隔离。"""
    global _log_context
    _log_context = isolated_agent_logs()
    _log_context.__enter__()


def tearDownModule():
    _log_context.__exit__(None, None, None)


class TestResearchTools(unittest.TestCase):
    """模型与检索IO隔离；原文、上传保存、加载和引用映射实际运行。"""

    def setUp(self):
        from src.data_loader import batch_import, create_import_tasks

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = deepcopy(load_config())
        self.config["paths"]["raw_documents"] = self.directory.name
        self.config["paths"]["logs"] = str(Path(self.directory.name) / "logs")
        self.addCleanup(patch.stopall)
        patch("src.agent.tools.load_config", return_value=self.config).start()
        patch("src.utils.logger.load_config", return_value=self.config).start()
        patch("src.agent.react_loop.load_config", return_value=self.config).start()
        self.text = ("AI Benchmark Study\nAuthors: Alice Smith, 张三\nPublished: 2024\nAbstract\n"
                     "This study evaluates AI models.\n本文比较人工智能模型，保留原始摘要。\nKeywords: AI\n"
                     "DOI: 10.1234/demo.2024\nReferences\nBob, Other Paper, 2020, 10.5678/other\n")
        tasks = create_import_tasks([("研究.md", self.text.encode())])
        list(batch_import(tasks, self.directory.name))
        self.path = Path(tasks[0]["path"])
        self.document = tasks[0]["documents"][0]
        self.doc_id = self.document.metadata["doc_id"]
        self.selection = {"title": "AI Benchmark Study", "authors": ["Alice Smith", "张三"], "year": 2024,
                          "doi": "10.1234/demo.2024"}
        self.response = {"model": "qwen2.5:7b", "done": True, "done_reason": "stop",
                         "prompt_eval_count": 100, "eval_count": 40,
                         "message": {"content": json.dumps(self.selection)}}

    def metadata(self, response=None, doc_id=None):
        raw = self.response if response is None else response
        with patch("src.generation.rag_pipeline.urlopen", return_value=BytesIO(json.dumps(raw).encode())) as http:
            result = paper_metadata.invoke({"doc_id": self.doc_id if doc_id is None else doc_id})
        return result, http

    def rag(self, scores, answer="结果见[参考文档1]。", doc_id=None):
        response = {**self.response, "message": {"content": answer}}
        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.generation.rag_pipeline.urlopen", return_value=BytesIO(json.dumps(response).encode())) as http:
            retriever.return_value.search.return_value = [(self.document, s) for s in scores]
            result = knowledge_base_search.invoke({"question": "模型比较结果是什么？", "doc_id": doc_id})
        return result, retriever, http

    def test_rag_stream_emits_before_tool_finishes_and_saves_only_final_answer(self):
        """核心RAG真实读取NDJSON，首包在末包之前进入会话事件，最终才保存整轮。"""
        from threading import Event
        from src.agent.memory import MemoryManager, run_session
        released, final_packet = Event(), Event()
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def __iter__(self):
                yield json.dumps({"message": {"content": "首段证据"}}).encode() + b"\n"
                released.wait(timeout=3)
                final_packet.set()
                yield json.dumps({**self.response, "message": {"content": "[参考文档1]。"}}).encode() + b"\n"
        response = Response()
        response.response = self.response
        memory = MemoryManager(Path(self.directory.name) / "sessions.sqlite3")
        session = memory.create_session("user")
        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.generation.streaming.urlopen", return_value=response), \
                patch("src.generation.rag_pipeline.generate_answer", side_effect=AssertionError("流式主链不能调用非流式生成")):
            retriever.return_value.search.return_value = [(self.document, .9)]
            events = run_session("知识库中模型的比较结果是什么？", "user", session,
                                 [knowledge_base_search], memory=memory, stream=True)
            try:
                first = next(event for event in events if event["type"] == "token")
                self.assertIn("首段证据", first["answer"])
                self.assertFalse(final_packet.is_set())
                self.assertEqual(memory.get_messages("user", session), [])
                released.set()
                rest = list(events)
                self.assertTrue(rest[-1]["task_complete"])
                self.assertEqual(memory.get_messages("user", session)[-1].content, rest[-1]["full_response"])
                self.assertIn("研究.md", rest[-1]["full_response"])
            finally:
                released.set()
                events.close()

    def test_registry_contains_real_tools_with_schemas(self):
        self.assertEqual([t.name for t in AVAILABLE_TOOLS], ["knowledge_base_search", "paper_metadata", "paper_compare", "keyword_extract", "paper_summary", "current_time", "calculator", "paper_list"])
        self.assertEqual(set(knowledge_base_search.args), {"question", "doc_id"})
        self.assertEqual(set(paper_metadata.args), {"doc_id"})
        self.assertIn("RAG", knowledge_base_search.description)
        self.assertIn("DOI", paper_metadata.description)

    def test_main_rag_stream_disconnect_keeps_partial_and_marks_incomplete(self):
        """主链未收到done时不能完成或覆盖部分正文，真实会话保存失败状态。"""
        from src.agent.memory import MemoryManager, run_session
        response = BytesIO(json.dumps({"message": {"content": "已收到的部分证据"}}).encode() + b"\n")
        memory = MemoryManager(Path(self.directory.name) / "broken-stream.sqlite3")
        session = memory.create_session("user")
        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.generation.streaming.urlopen", return_value=response):
            retriever.return_value.search.return_value = [(self.document, .9)]
            events = list(run_session("知识库中模型的比较结果是什么？", "user", session,
                                      [knowledge_base_search], memory=memory, stream=True))
        final = events[-1]
        self.assertFalse(final["task_complete"])
        self.assertIn("已收到的部分证据", final["full_response"])
        self.assertIn("未完成", final["full_response"])
        self.assertFalse(memory.get_messages("user", session)[-1].additional_kwargs["task_complete"])
        self.assertTrue(response.closed)

    def test_rag_uses_hybrid_rerank_and_real_module_two_citations(self):
        result, retriever, http = self.rag([0.9])
        retriever.return_value.search.assert_called_once_with("模型比较结果是什么？", doc_id=None, rerank=True)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["generation_mode"], "grounded")
        self.assertEqual(result["citations"][0]["metadata"]["doc_id"], self.doc_id)
        self.assertIn("研究.md", result["answer"])
        self.assertIn("行1", result["answer"])
        self.assertEqual(result["usage"]["eval_count"], 40)
        self.assertIn("AI Benchmark Study", json.loads(http.call_args.args[0].data)["messages"][1]["content"])

    def test_rag_low_relevance_returns_candidates_without_generation(self):
        result, _, http = self.rag([0.01])
        self.assertEqual(result["status"], "needs_confirmation")
        self.assertEqual(result["references"][0]["metadata"]["doc_id"], self.doc_id)
        self.assertEqual(result["usage"], {"prompt_eval_count": 0, "eval_count": 0})
        http.assert_not_called()

    def test_rag_tool_keeps_sources_without_repeating_large_pdf_layout(self):
        self.document.metadata["formula_layout"] = "原文坐标" * 10000
        result, _, _ = self.rag([0.9])
        self.assertNotIn("formula_layout", result["citations"][0]["metadata"])
        self.assertEqual(result["citations"][0]["metadata"]["doc_id"], self.doc_id)
        self.assertIn("formula_layout", self.document.metadata)

    def test_rag_tool_logs_success_confirmation_and_model_error_with_real_usage(self):
        from src.utils.logger import read_rag_requests

        self.rag([0.9])
        self.rag([0.01])
        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.generation.rag_pipeline.urlopen", side_effect=URLError("模型断开")):
            retriever.return_value.search.return_value = [(self.document, 0.9)]
            execute_tool("knowledge_base_search", {"question": "问题"}, AVAILABLE_TOOLS)
        records, invalid = read_rag_requests()
        self.assertEqual(invalid, 0)
        self.assertEqual([r["status"] for r in records], ["completed", "awaiting_confirmation", "error"])
        self.assertEqual(records[0]["tokens"]["input"], 100)
        self.assertEqual(records[1]["tokens"]["input"], 0)
        self.assertIsNone(records[2]["tokens"]["input"])
        self.assertEqual(records[0]["retrieval"]["documents"][0]["metadata"]["doc_id"], self.doc_id)

    def test_rag_logging_failure_warns_without_discarding_actual_answer(self):
        with patch("src.utils.logger.record_rag_request", side_effect=OSError("不可写")):
            result, _, _ = self.rag([0.9])
        self.assertEqual(result["status"], "answered")
        self.assertIn("研究.md", result["answer"])
        self.assertIn("日志保存失败", result["warnings"][-1])

    def test_recovered_vector_notice_is_kept_for_answers_and_pending_candidates(self):
        """正常答案和低相关确认都显示索引恢复提示；低相关仍不启动生成。"""
        notice = "本次临时计算指定文档向量，未修改原索引。"
        self.document.metadata["retrieval_warning"] = notice
        for score in (.9, .01):
            with self.subTest(score=score):
                result, _, http = self.rag([score])
                self.assertEqual(result["warnings"], [notice])
                self.assertEqual(http.call_count, 1 if score == .9 else 0)

    def test_rag_empty_library_preserves_explicit_fallback_notice(self):
        result, _, http = self.rag([], "没有资料，不能说明该论文的结果。")
        self.assertEqual(result["generation_mode"], "empty")
        self.assertIn("当前知识库中未找到相关文档", result["answer"])
        self.assertEqual(result["citations"], [])
        self.assertEqual(http.call_count, 1)

    def test_rag_validates_question_and_uploaded_document_before_retrieval(self):
        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever:
            for args in ({"question": " "}, {"question": "问题", "doc_id": "../private"},
                         {"question": "问题", "doc_id": "0" * 64}):
                with self.subTest(args=args):
                    result = execute_tool("knowledge_base_search", args, AVAILABLE_TOOLS)
                    self.assertEqual(result["status"], "error")
            retriever.assert_not_called()

    def test_rag_filters_by_existing_uploaded_fingerprint(self):
        result, retriever, _ = self.rag([0.9], doc_id=self.doc_id)
        self.assertEqual(result["doc_id"], self.doc_id)
        self.assertEqual(retriever.return_value.search.call_args.kwargs["doc_id"], self.doc_id)

    def test_generation_failure_is_real_tool_error_without_retry(self):
        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.generation.rag_pipeline.urlopen", side_effect=URLError("模型断开")) as http:
            retriever.return_value.search.return_value = [(self.document, 0.9)]
            result = execute_tool("knowledge_base_search", {"question": "模型结果？"}, AVAILABLE_TOOLS)
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["result"])
        self.assertIn("Ollama", result["error"])
        self.assertEqual(http.call_count, 1)

    def test_metadata_returns_original_fields_and_real_evidence(self):
        result, http = self.metadata()
        self.assertEqual(result["title"], "AI Benchmark Study")
        self.assertEqual(result["authors"], ["Alice Smith", "张三"])
        self.assertEqual(result["year"], 2024)
        self.assertEqual(result["doi"], "10.1234/demo.2024")
        self.assertEqual(result["abstract"], "This study evaluates AI models.\n本文比较人工智能模型，保留原始摘要。")
        self.assertEqual(result["evidence"]["abstract"][0]["location"], "行5")
        self.assertIn("张三", result["evidence"]["authors"][1]["text"])
        self.assertEqual(result["missing_fields"], [])
        self.assertEqual(self.path.read_text(), self.text)
        payload = json.loads(http.call_args.args[0].data)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["format"]["properties"]["doi"]["enum"], [None, "10.1234/demo.2024", "10.5678/other"])
        self.assertEqual(payload["options"]["num_predict"], self.config["llm"]["num_predict"])

    def test_metadata_inline_abstract_punctuation_keeps_original_body(self):
        # 模拟DETR版式的同行摘要，来源位置仍取加载器的真实行号。
        rows = [{"text": text, "location": f"行{i}", "metadata": {}}
                for i, text in enumerate(["Abstract. We present an end-to-end detector.",
                                          "It uses a transformer.", "1 Introduction", "Not abstract."], 1)]
        empty = {"title": None, "authors": [], "year": None, "doi": None}
        for heading in ("Abstract. ", "Abstract: ", "Abstract—", "摘要："):
            rows[0]["text"] = heading + "We present an end-to-end detector."
            with patch("src.agent.tools._metadata_lines", return_value=(rows, False)):
                result, _ = self.metadata({**self.response, "message": {"content": json.dumps(empty)}})
            self.assertEqual(result["abstract"], "We present an end-to-end detector.\nIt uses a transformer.")
            self.assertEqual(result["evidence"]["abstract"][0]["location"], "行1")
            self.assertNotIn("abstract", result["missing_fields"])

    def test_metadata_abstract_word_inside_sentence_is_not_a_heading(self):
        rows = [{"text": "This abstract discusses AI.", "location": "行1", "metadata": {}}]
        empty = {"title": None, "authors": [], "year": None, "doi": None}
        with patch("src.agent.tools._metadata_lines", return_value=(rows, False)):
            result, _ = self.metadata({**self.response, "message": {"content": json.dumps(empty)}})
        self.assertIsNone(result["abstract"])

    def test_metadata_missing_fields_are_null_not_guessed_from_filename(self):
        selection = {"title": None, "authors": [], "year": None, "doi": None}
        result, _ = self.metadata({**self.response, "message": {"content": json.dumps(selection)}})
        self.assertEqual(set(result["missing_fields"]), set(selection))
        self.assertIsNone(result["title"])
        self.assertEqual(result["authors"], [])
        self.assertEqual(result["evidence"]["title"], [])

    def test_metadata_hallucinated_author_year_or_doi_is_rejected(self):
        for field, value in (("authors", ["Invented Person"]),
                             ("year", 2025),
                             ("doi", "10.1234/invented")):
            selection = {**self.selection, field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "不在所选原文"):
                self.metadata({**self.response, "message": {"content": json.dumps(selection)}})

    def test_metadata_rejects_invalid_types_ranges_and_extra_fields(self):
        selections = [{**self.selection, "title": "改写标题"}, {**self.selection, "authors": None},
                      {**self.selection, "authors": [None]}, {**self.selection, "extra": "多余字段"},
                      {**self.selection, "year": True}, {**self.selection, "year": 1},
                      {**self.selection, "abstract": "不允许模型生成的字段"}]
        for selection in selections:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.metadata({**self.response, "message": {"content": json.dumps(selection)}})

    def test_metadata_unfinished_or_malformed_model_results_do_not_return_partial_fields(self):
        for response in ({**self.response, "done_reason": "length"}, {**self.response, "done": False},
                         {"error": "模型失败"}, {**self.response, "model": None},
                         {**self.response, "message": {"content": "不是JSON"}}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                self.metadata(response)

    def test_unknown_fingerprint_stale_file_and_outside_symlink_never_call_model(self):
        with patch("src.generation.rag_pipeline.urlopen") as http:
            for identifier in ("论文A", "../secret", "0" * 64):
                with self.subTest(identifier=identifier), self.assertRaises((ValueError, FileNotFoundError)):
                    paper_metadata.invoke({"doc_id": identifier})
            self.path.write_text("原文已修改")
            with self.assertRaisesRegex(ValueError, "指纹不一致"):
                paper_metadata.invoke({"doc_id": self.doc_id})
            self.path.unlink()
            outside = Path(self.directory.name).parent / (Path(self.directory.name).name + "-outside.md")
            outside.write_text(self.text)
            self.addCleanup(outside.unlink)
            self.path.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "路径或内容指纹"):
                paper_metadata.invoke({"doc_id": self.doc_id})
            http.assert_not_called()

    def test_metadata_reuses_pdf_and_word_loaders_with_real_locations(self):
        import pymupdf
        from docx import Document as WordDocument
        from src.data_loader import batch_import, create_import_tasks

        sample = "AI Study\nAuthors: Alice\nPublished: 2024\nAbstract\nReal abstract text.\nKeywords: AI\nDOI: 10.1234/study"
        pdf = pymupdf.open()
        pdf.new_page().insert_text((72, 72), sample)
        pdf_bytes = pdf.tobytes()
        pdf.close()
        word = WordDocument()
        for line in sample.splitlines():
            word.add_paragraph(line)
        buffer = BytesIO()
        word.save(buffer)
        for name, data, location in (("论文.pdf", pdf_bytes, "第1页（物理页码）"),
                                     ("论文.docx", buffer.getvalue(), "段落5")):
            tasks = create_import_tasks([(name, data)])
            list(batch_import(tasks, self.directory.name))
            selection = {"title": "AI Study", "authors": ["Alice"], "year": 2024,
                         "doi": "10.1234/study"}
            result, _ = self.metadata({**self.response, "message": {"content": json.dumps(selection)}},
                                      tasks[0]["documents"][0].metadata["doc_id"])
            self.assertEqual(result["abstract"], "Real abstract text.")
            self.assertEqual(result["evidence"]["abstract"][0]["location"], location)

    def test_metadata_character_budget_preserves_contiguous_prefix(self):
        from src.agent.tools import _metadata_lines

        self.config["generation"]["max_context_chars"] = len(self.text.splitlines()[0]) + 1
        lines, truncated = _metadata_lines(self.path)
        self.assertEqual([row["text"] for row in lines], ["AI Benchmark Study"])
        self.assertTrue(truncated)

    def test_metadata_unknown_usage_stays_unknown_and_cloud_is_rejected(self):
        response = deepcopy(self.response)
        del response["prompt_eval_count"], response["eval_count"]
        result, http = self.metadata(response)
        self.assertEqual(result["usage"], {"prompt_eval_count": None, "eval_count": None})
        self.assertEqual(http.call_count, 1)
        self.config["llm"]["base_url"] = "http://example.com"
        with patch("src.generation.rag_pipeline.urlopen") as http, self.assertRaises(ValueError):
            paper_metadata.invoke({"doc_id": self.doc_id})
        http.assert_not_called()

    def test_default_agent_can_execute_registered_rag_and_preserve_citations(self):
        def packet(content):
            return {**self.response, "message": {"content": json.dumps(content)}}

        plan = {"thought": "先检索论文。", "next_step": "tool", "tool_name": "knowledge_base_search"}
        action = {**self.response, "message": {"tool_calls": [{"function": {
            "name": "knowledge_base_search", "arguments": {"question": "模型结果？"}}}]}}
        observed = {"observation": "已获得带来源答案。", "decision": "finish", "task_complete": True,
                    "answer": "见研究.md的原文。"}
        rag_response = {**self.response, "message": {"content": "结果见[参考文档1]。"}}
        with patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.generation.rag_pipeline.urlopen", return_value=BytesIO(json.dumps(rag_response).encode())), \
                patch("src.agent.react_loop.urlopen", side_effect=[BytesIO(json.dumps(r).encode()) for r in
                                                                   (packet(plan), action, packet(observed))]):
            retriever.return_value.search.return_value = [(self.document, 0.9)]
            events = list(run_react("模型结果？"))
        self.assertTrue(events[-1]["task_complete"])
        self.assertEqual(events[1]["name"], "knowledge_base_search")
        self.assertEqual(events[-1]["context"]["observations"][0]["result"]["citations"][0]["source_file"], "研究.md")

    def test_unconfirmed_rag_candidates_cannot_be_marked_complete(self):
        context = {"observations": [{"status": "success", "result": {"status": "needs_confirmation"}}]}
        decision = {"observation": "等待确认。", "decision": "finish", "task_complete": True, "answer": "假称完成"}
        with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps({
            **self.response, "message": {"content": json.dumps(decision)}}).encode())), self.assertRaisesRegex(RuntimeError, "尚待用户确认"):
            observe("问题", AVAILABLE_TOOLS, context)


if __name__ == "__main__":
    unittest.main()
