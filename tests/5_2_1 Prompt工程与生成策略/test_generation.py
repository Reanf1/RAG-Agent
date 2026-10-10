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
from urllib.error import URLError
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
                       "generation": {"max_context_chars": 6000, "max_prompt_chars": 12000,
                                      "low_relevance_threshold": 0.1}}
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

    def test_token_overflow_is_rejected_before_network(self):
        """稀有Unicode字符的Token数远大于字符数，最终请求不能照常放行。"""
        from src.generation.rag_pipeline import _build_generation_request
        with self.assertRaisesRegex(ValueError, "Token"):
            _build_generation_request("预算检查？", {"context": "🧬" * 6000, "references": []})
        self.opener.assert_not_called()


    def test_network_failure_is_explicit_without_fallback(self):
        self.opener.side_effect = URLError("本机服务不可用")
        with self.assertRaisesRegex(RuntimeError, "本地 Ollama 调用失败"):
            generate_answer("层数？", self.context)
        self.assertEqual(self.opener.call_count, 1)


if __name__ == "__main__":
    unittest.main()
