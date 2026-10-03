"""5.3.2 工具集开发：TestCalculatorAndPaperList。"""

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
from src.agent.react_loop import run_react
from src.agent.tools import AVAILABLE_TOOLS, execute_tool
from src.agent.tools import calculator, paper_list, web_search
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


class TestCalculatorAndPaperList(unittest.TestCase):
    """真实算式/上传/Chroma操作；仅用小型测试向量和构造模型HTTP隔离外部推理。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = deepcopy(load_config())
        self.config["paths"]["raw_documents"] = str(Path(self.directory.name) / "raw")
        self.config["paths"]["vector_index"] = str(Path(self.directory.name) / "index")
        for module in ("src.agent.tools", "src.agent.router", "src.retrieval.vector_store"):
            patcher = patch(module + ".load_config", return_value=self.config)
            patcher.start()
            self.addCleanup(patcher.stop)

    def upload(self, name="研究.md", text="DO_NOT_RETURN_BODY：Transformer研究。"):
        from src.data_loader import batch_import, create_import_tasks
        tasks = create_import_tasks([(name, text.encode())])
        list(batch_import(tasks, self.config["paths"]["raw_documents"]))
        self.assertEqual(tasks[0]["status"], "success")
        return tasks[0]

    def index(self, tasks):
        from src.chunking import split_documents
        from src.retrieval.vector_store import VectorStore
        from tests.helpers import SmallEmbeddings
        self.embeddings = SmallEmbeddings()
        store = VectorStore(embeddings=self.embeddings)
        for task in tasks:
            store.add_chunks(split_documents(task["documents"]))
        return store

    def test_registry_has_eight_unique_local_tools_with_real_schemas(self):
        self.assertEqual(len(AVAILABLE_TOOLS), 8)
        self.assertEqual(len({item.name for item in AVAILABLE_TOOLS}), 8)
        self.assertIn(calculator, AVAILABLE_TOOLS)
        self.assertIn(paper_list, AVAILABLE_TOOLS)
        self.assertEqual(set(calculator.args), {"expression"})
        self.assertEqual(paper_list.args, {})
        self.assertNotIn(web_search, AVAILABLE_TOOLS)

    def test_calculator_decimal_accuracy_and_precedence(self):
        for expression, expected in (("3.14*2.56", "8.0384"), ("0.1+0.2", "0.3"),
                                     ("(1+2)*3-4/2", "7"), ("1/3", "0.3333333333333333333333333333")):
            with self.subTest(expression=expression):
                self.assertEqual(calculator.invoke({"expression": expression})["result"], expected)

    def test_calculator_negative_unary_chinese_and_symbols(self):
        for expression, expected in (("-2*(-3)+(+1)", "7"), ("3.14乘以2.56", "8.0384"),
                                     ("8÷2加1×3减2", "5"), ("-0.0000", "0")):
            with self.subTest(expression=expression):
                self.assertEqual(calculator.invoke({"expression": expression})["result"], expected)

    def test_calculator_rejects_non_arithmetic_and_unsupported_operators(self):
        for expression in ("__import__('os').system('bad')", "True", "abs(-1)", "1e100000000", "2**3", "4//2", "3%2", "[1]", "1,2", "1j"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                calculator.invoke({"expression": expression})

    def test_calculator_empty_syntax_length_and_depth_limits(self):
        for expression in (" ", "1+", "(1+2", "1" * 257, "-" * 70 + "1"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                calculator.invoke({"expression": expression})

    def test_calculator_zero_division_is_real_input_error(self):
        event = execute_tool("calculator", {"expression": "1/(2-2)"}, AVAILABLE_TOOLS)
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["error_kind"], "input")
        self.assertIn("不能除以零", event["error"])
        self.assertIsNone(event["result"])

    def test_utility_tools_validate_arguments_without_model_or_network(self):
        with patch("socket.create_connection", side_effect=AssertionError("不应联网")), patch("src.agent.react_loop.urlopen") as http:
            result = calculator.invoke({"expression": "0.1+0.2"})
            listed = paper_list.invoke({})
            for name, arguments in (("calculator", {}), ("calculator", {"expression": 12}),
                                    ("calculator", {"expression": "1+2", "path": "其他路径"}), ("paper_list", {"doc_id": "其他文档"})):
                self.assertEqual(execute_tool(name, arguments, AVAILABLE_TOOLS)["status"], "error")
        self.assertEqual(result["usage"], {"prompt_eval_count": 0, "eval_count": 0})
        self.assertEqual(listed["usage"], result["usage"])
        http.assert_not_called()

    def test_empty_library_does_not_create_raw_directory_or_chroma(self):
        with patch("src.retrieval.vector_store.VectorStore", side_effect=AssertionError("不应创建空库")):
            result = paper_list.invoke({})
        self.assertEqual(result["papers"], [])
        self.assertEqual(result["total"], 0)
        self.assertFalse(Path(self.config["paths"]["raw_documents"]).exists())
        self.assertFalse(Path(self.config["paths"]["vector_index"]).exists())

    def test_uploaded_not_indexed_paper_has_usable_id_and_name(self):
        task = self.upload()
        result = paper_list.invoke({})
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["papers"][0], {"doc_id": task["documents"][0].metadata["doc_id"],
                         "source_file": "研究.md", "indexed_chunks": 0, "source_available": True, "index_status": "not_indexed"})
        self.assertNotIn("DO_NOT_RETURN_BODY", json.dumps(result, ensure_ascii=False))

    def test_real_index_counts_chunks_without_loading_embedding_or_querying(self):
        task = self.upload(text="人工构造的论文正文。" * 100)
        store = self.index([task])
        calls = len(self.embeddings.document_calls)
        with patch("src.retrieval.vector_store.get_embeddings", side_effect=AssertionError("列表不加载模型")), \
                patch("src.retrieval.vector_store.VectorStore.search", side_effect=AssertionError("列表不检索")):
            result = paper_list.invoke({})
        row = result["papers"][0]
        self.assertEqual(row["indexed_chunks"], store.count())
        self.assertGreater(row["indexed_chunks"], 1)
        self.assertEqual(row["index_status"], "has_index")
        self.assertEqual(len(self.embeddings.document_calls), calls)

    def test_partial_index_means_has_chunks_not_complete_import(self):
        from src.chunking import split_documents
        task = self.upload(text="人工构造的论文正文。" * 100)
        chunks = split_documents(task["documents"])
        store = self.index([])
        store.add_chunks(chunks[:1])
        row = paper_list.invoke({})["papers"][0]
        self.assertGreater(len(chunks), 1)
        self.assertEqual(row["indexed_chunks"], 1)
        self.assertEqual(row["index_status"], "has_index")
        self.assertNotIn("indexed", row)  # 不用布尔值冒充全部预期块完成。

    def test_duplicate_content_with_alias_names_is_one_canonical_paper(self):
        first = self.upload(name="Z.md")
        self.upload(name="A.md")
        self.index([first])
        result = paper_list.invoke({})
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["papers"][0]["source_file"], "A.md")
        self.assertEqual(result["papers"][0]["indexed_chunks"], 1)

    def test_same_name_different_content_keeps_distinct_paper_ids(self):
        first, second = self.upload(text="论文A"), self.upload(text="论文B")
        result = paper_list.invoke({})
        self.assertEqual(result["total"], 2)
        self.assertEqual({row["doc_id"] for row in result["papers"]},
                         {task["documents"][0].metadata["doc_id"] for task in (first, second)})

    def test_missing_original_keeps_index_record_with_explicit_status(self):
        task = self.upload()
        self.index([task])
        Path(task["path"]).unlink()
        row = paper_list.invoke({})["papers"][0]
        self.assertEqual(row["index_status"], "source_missing")
        self.assertFalse(row["source_available"])
        self.assertEqual(row["indexed_chunks"], 1)

    def test_index_deletion_updates_list_and_preserves_uploaded_source(self):
        task = self.upload()
        store = self.index([task])
        store.delete_document(task["documents"][0].metadata["doc_id"])
        row = paper_list.invoke({})["papers"][0]
        self.assertEqual(row["indexed_chunks"], 0)
        self.assertEqual(row["index_status"], "not_indexed")
        self.assertTrue(Path(task["path"]).is_file())

    def test_source_corruption_and_symlink_escape_are_not_fake_empty_lists(self):
        task = self.upload()
        path = Path(task["path"])
        original = path.read_bytes()
        path.write_text("被改动的原文")
        with self.assertRaises(ValueError):
            paper_list.invoke({})
        path.unlink()
        outside = Path(self.directory.name) / "outside.md"
        outside.write_bytes(original)
        path.symlink_to(outside)
        with self.assertRaises(ValueError):
            paper_list.invoke({})

    def test_unrelated_raw_files_are_not_uploaded_papers(self):
        root = Path(self.config["paths"]["raw_documents"])
        root.mkdir()
        (root / "未导入.md").write_text("资料准备文件，不是成功上传记录")
        folder = root / ("a" * 64)
        folder.mkdir()
        (folder / "unsupported.bin").write_bytes(b"test")
        self.assertEqual(paper_list.invoke({})["total"], 0)

    def test_index_read_error_propagates_instead_of_reporting_empty_library(self):
        self.index([self.upload()])
        with patch("src.retrieval.vector_store.VectorStore.list_chunks", side_effect=RuntimeError("明确注入读取故障")):
            event = execute_tool("paper_list", {}, AVAILABLE_TOOLS)
        self.assertEqual(event["status"], "error")
        self.assertIn("读取故障", event["error"])

    def test_real_tools_route_correctly_and_paper_fact_queries_still_use_rag(self):
        for question, name in (("3.14乘以2.56", "calculator"), ("请计算(1+2)*3", "calculator"),
                               ("列出已上传论文", "paper_list"), ("文献列表", "paper_list"),
                               ("有哪些论文？", "paper_list"), ("list uploaded papers", "paper_list")):
            self.assertEqual(route_question(question, AVAILABLE_TOOLS)["tool_name"], name)
        self.assertEqual(route_question("哪些论文使用Transformer？", AVAILABLE_TOOLS)["tool_name"], "knowledge_base_search")
        self.assertEqual(route_question("列出这篇论文的实验结果", AVAILABLE_TOOLS)["tool_name"], "knowledge_base_search")
        self.assertIsNone(route_question("文献列表", []))
        self.assertIsNone(route_question("文献列表", AVAILABLE_TOOLS, {"summary": "已有会话摘要"}))

    def test_agent_action_really_executes_both_registered_utility_tools(self):
        task = self.upload()
        for name, question, arguments, answer in (("calculator", "3.14乘以2.56", {"expression": "3.14*2.56"}, "8.0384"),
                                                  ("paper_list", "列出已上传论文", {}, "研究.md")):
            action = {"model": "qwen2.5:7b", "done": True, "done_reason": "stop", "message": {
                      "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}}
            observation = {**action, "message": {"content": json.dumps({"observation": "实际工具已返回结果。",
                          "decision": "finish", "task_complete": True, "answer": answer})}}
            with self.subTest(name=name), patch("src.agent.react_loop.urlopen", side_effect=[
                    BytesIO(json.dumps(action).encode()), BytesIO(json.dumps(observation).encode())]) as http:
                events = list(run_react(question))
            result = next(event for event in events if event["type"] == "tool_result")
            self.assertEqual(result["status"], "success")
            self.assertEqual(events[0]["route"], "rule")
            self.assertTrue(events[-1]["task_complete"])
            self.assertEqual(http.call_count, 2)
            if name == "calculator":
                self.assertEqual(result["result"]["result"], answer)
            else:
                self.assertEqual(result["result"]["papers"][0]["doc_id"], task["documents"][0].metadata["doc_id"])


if __name__ == "__main__":
    unittest.main()
