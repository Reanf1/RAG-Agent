"""5.2.1 Prompt工程与生成策略：TestLocalGeneration、TestGenerationEvaluation。"""

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
from urllib.error import HTTPError, URLError
from langchain_core.documents import Document
from src.generation.rag_pipeline import build_context, generate_answer
from importlib import import_module

# 按完整模块名加载含编号和空格的小节评测函数。
evaluate_answer = import_module("reports.5_2_1 Prompt工程与生成策略.compare_generation").evaluate_answer
summarize = import_module("reports.5_2_1 Prompt工程与生成策略.compare_generation").summarize


class TestLocalGeneration(unittest.TestCase):
    """核对实际 HTTP 请求参数、失败边界与同轮引用；mock 不能证明答案质量。"""

    def setUp(self):
        self.config = {"llm": {"provider": "ollama", "base_url": "http://localhost:11434",
                              "model": "qwen2.5:7b", "temperature": 0.1, "top_p": 0.9,
                              "top_k": 40, "num_ctx": 8192, "num_predict": 512,
                              "repeat_penalty": 1.0},
                       "generation": {"max_context_chars": 6000, "max_prompt_chars": 12000}}
        config_patch = patch("src.generation.rag_pipeline.load_config", return_value=self.config)
        config_patch.start()
        self.addCleanup(config_patch.stop)
        self.context = build_context("层数？", [(Document(page_content="编码器有6层。", metadata={
            "source_file": "attention.pdf", "page_number": 3, "chunk_id": "真实测试块"}), 1.0)])
        self.response = {"model": "qwen2.5:7b", "done": True, "done_reason": "stop",
                         "message": {"role": "assistant", "content":
                                     "## 回答\n编码器有6层。[参考文档1]\n## 参考来源\n[参考文档1]"},
                         "eval_count": 30, "prompt_eval_count": 400, "total_duration": 1000000000}
        opener_patch = patch("src.generation.rag_pipeline.urlopen")
        self.opener = opener_patch.start()
        self.addCleanup(opener_patch.stop)
        self.set_response(self.response)

    def set_response(self, response):
        self.opener.return_value = BytesIO(json.dumps(response).encode())

    def test_config_reaches_native_ollama_and_citations(self):
        before = deepcopy(self.context)
        result = generate_answer("层数？", self.context)
        request = self.opener.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(payload["options"]["top_k"], 40)
        self.assertEqual(payload["options"]["temperature"], 0.1)
        self.assertEqual(payload["options"]["top_p"], 0.9)
        self.assertEqual(payload["model"], "qwen2.5:7b")
        self.assertFalse(payload["stream"])
        self.assertEqual([message["role"] for message in payload["messages"]], ["system", "user"])
        self.assertIn("attention.pdf；第3页", result["answer"])
        self.assertEqual(result["raw_answer"], self.response["message"]["content"])
        self.assertEqual(result["usage"]["eval_count"], 30)
        self.assertEqual(self.context, before)

    def test_experiment_options_do_not_mutate_config(self):
        options = {"temperature": 0.8, "seed": 17}
        before = deepcopy(self.config)
        result = generate_answer("层数？", self.context, options=options)
        self.assertEqual(result["options"]["temperature"], 0.8)
        self.assertEqual(result["options"]["seed"], 17)
        self.assertEqual(self.config, before)
        self.assertEqual(options, {"temperature": 0.8, "seed": 17})

    def test_invalid_sampling_is_rejected_before_network(self):
        for options in ({"temperature": -1}, {"temperature": float("nan")}, {"top_p": 0},
                        {"top_p": 1.1}, {"top_k": True}, {"top_k": 0}, {"num_ctx": -1},
                        {"num_predict": -1}, {"repeat_penalty": 0}, {"seed": "17"}, {"unknown": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                generate_answer("层数？", self.context, options=options)
        self.opener.assert_not_called()

    def test_cloud_and_non_http_configuration_is_rejected(self):
        for url in ("https://example.com", "http://example.com", "file:///tmp/model",
                    "http://localhost:11434?remote=1"):
            self.config["llm"]["base_url"] = url
            with self.subTest(url=url), self.assertRaises(ValueError):
                generate_answer("层数？", self.context)
        self.opener.assert_not_called()

    def test_network_failure_is_explicit_without_fallback(self):
        self.opener.side_effect = URLError("本机服务不可用")
        with self.assertRaisesRegex(RuntimeError, "本地 Ollama 调用失败"):
            generate_answer("层数？", self.context)
        self.assertEqual(self.opener.call_count, 1)

    def test_blank_or_unfinished_response_is_rejected(self):
        for response in ({**self.response, "done": False}, {**self.response, "message": {"content": " "}},
                         {**self.response, "error": "模型不可用"}):
            self.set_response(response)
            expected = "生成失败" if response.get("error") else "非空答案"
            with self.subTest(response=response), self.assertRaisesRegex(RuntimeError, expected):
                generate_answer("层数？", self.context)

    def test_length_stop_is_preserved_for_evaluation(self):
        self.set_response({**self.response, "done_reason": "length"})
        result = generate_answer("层数？", self.context)
        self.assertEqual(result["done_reason"], "length")
        self.assertIn("回答已达到生成 Token 上限，内容可能尚未完整。", result["warnings"])

    def test_empty_question_does_not_call_network(self):
        with self.assertRaisesRegex(ValueError, "问题不能为空"):
            generate_answer(" ", self.context)
        self.opener.assert_not_called()

    def test_empty_fallback_is_explicit_even_when_model_omits_notice(self):
        context = build_context("什么是注意力？", [])
        self.set_response({**self.response, "message": {"content": "注意力是一种加权汇总机制。"}})
        result = generate_answer("什么是注意力？", context)
        self.assertTrue(result["answer"].startswith("当前知识库中未找到相关文档。"))
        self.assertIn("纯模型回答", result["answer"])
        self.assertEqual(result["generation_mode"], "empty")
        self.assertEqual(result["citations"], [])
        self.assertIn("模型自身知识", json.loads(self.opener.call_args.args[0].data)["messages"][0]["content"])

    def test_http_failure_has_specific_advice_and_original_cause(self):
        for code, expected in ((404, "ollama list"), (400, "生成参数"), (500, "可用内存")):
            failure = HTTPError("http://localhost:11434/api/chat", code, "failed", {}, BytesIO(b'{"error":"local error"}'))
            self.opener.side_effect = failure
            with self.assertRaisesRegex(RuntimeError, expected) as raised:
                generate_answer("层数？", self.context)
            self.assertIs(raised.exception.__cause__, failure)
            self.assertTrue(failure.closed)


class TestGenerationEvaluation(unittest.TestCase):
    """评分规则刻意只称覆盖检查，语义正确性仍需核对实际原文和答案。"""

    def setUp(self):
        self.case = {"id": "test", "fact_patterns": [r"\b6\b", r"编码器"],
                     "expected_citation_ids": [1]}
        self.result = {"raw_answer": "## 回答\n编码器有6层。[参考文档1]\n## 参考来源\n[参考文档1]",
                       "citations": [{"id": 1}], "invalid_citation_ids": [], "done_reason": "stop"}

    def test_complete_answer_passes_and_footer_does_not_fill_facts(self):
        self.assertTrue(evaluate_answer(self.case, self.result)["rule_pass"])
        self.result["raw_answer"] = "## 回答\n未知\n## 参考来源\n编码器有6层。[参考文档1]"
        self.assertEqual(evaluate_answer(self.case, self.result)["fact_coverage"], 0)

    def test_missing_expected_and_invalid_citations_fail(self):
        for citations, invalid in (([], []), ([{"id": 2}], []), ([{"id": 1}], [99])):
            self.result.update(citations=citations, invalid_citation_ids=invalid)
            self.assertFalse(evaluate_answer(self.case, self.result)["rule_pass"])

    def test_incomplete_output_is_not_a_pass(self):
        self.result["done_reason"] = "length"
        checks = evaluate_answer(self.case, self.result)
        self.assertFalse(checks["natural_stop"])
        self.assertFalse(checks["rule_pass"])

    def test_english_source_footer_is_not_a_body_citation(self):
        self.result["raw_answer"] = "## Answer\n编码器有6层。\n## Reference Sources\n[参考文档1]"
        checks = evaluate_answer(self.case, self.result)
        self.assertFalse(checks["expected_citations_ok"])
        self.assertFalse(checks["format_ok"])

    def test_template_no_document_notice_counts_as_refusal(self):
        case = {"id": "insufficient", "fact_patterns": ["无法回答"], "expected_citation_ids": []}
        self.result["raw_answer"] = "## 回答\n当前知识库中未找到相关文档。\n## 参考来源\n无可引用来源"
        self.result["citations"] = []
        self.assertTrue(evaluate_answer(case, self.result)["rule_pass"])

    def test_equal_weight_summary_with_two_seeds(self):
        checks = evaluate_answer(self.case, self.result)
        # 小统计样例同时包含事实题和拒答题，不调用模型。
        rows = [{"seed": seed, "wall_seconds": seconds,
                 "checks": {**checks, "citation_required": seed == 17},
                 "result": {"usage": {"eval_count": tokens}}}
                for seed, seconds, tokens in ((17, 2, 30), (29, 4, 50))]
        summary = summarize(rows)
        self.assertEqual(summary["wall_seconds_mean"], 3)
        self.assertEqual(summary["output_tokens_mean"], 40)
        self.assertEqual(summary["rule_pass"], 1)


if __name__ == "__main__":
    unittest.main()
