"""5.2.3 缓存与降级策略：TestDegradationContext。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from threading import Thread
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from src.generation.rag_pipeline import generate_answer, prepare_rag_context
from src.generation.streaming import stream_answer


class TestDegradationContext(unittest.TestCase):
    """使用确定分数验证策略边界；分数不是实际检索质量的标注。"""

    def setUp(self):
        from src.utils.config import load_config
        self.config = load_config()
        patcher = patch("src.generation.rag_pipeline.load_config", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.doc = Document(page_content="论文原文。", metadata={"source_file": "a.pdf", "page_number": 2})

    def test_empty_and_blank_documents_use_pure_model_mode(self):
        for results in ([], [(Document(page_content=" \n"), 0.9)]):
            context = prepare_rag_context("什么是注意力？", results)
            self.assertEqual(context["generation_mode"], "empty")
            self.assertIsNone(context["top_score"])
            self.assertEqual(context["references"], [])

    def test_low_score_boundary_and_unsorted_top_score(self):
        for score, mode in ((0, "low"), (0.099, "low"), (0.1, "grounded"), (1, "grounded")):
            context = prepare_rag_context("问题", [(self.doc, score)])
            self.assertEqual(context["generation_mode"], mode)
            self.assertEqual(context["top_score"], score)
            self.assertFalse(context["confirmed"])
            self.assertEqual(context["references"][0]["text"], self.doc.page_content)
        context = prepare_rag_context("问题", [(self.doc, 0.01), (self.doc, 0.8)])
        self.assertEqual(context["generation_mode"], "grounded")

    def test_invalid_scores_and_threshold_never_become_empty_results(self):
        for score in (-1, 1.2, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                prepare_rag_context("问题", [(self.doc, score)])
        for threshold in (0, 1, True, float("nan")):
            self.config["generation"]["low_relevance_threshold"] = threshold
            with self.assertRaises(ValueError):
                prepare_rag_context("问题", [])

    def test_unconfirmed_low_context_cannot_call_either_model_interface(self):
        context = prepare_rag_context("问题", [(self.doc, 0.01)])
        with patch("src.generation.rag_pipeline.urlopen") as sync, patch("src.generation.streaming.urlopen") as stream:
            with self.assertRaisesRegex(ValueError, "先查看"):
                generate_answer("问题", context)
            event = list(stream_answer("问题", context))[-1]
            self.assertEqual(event["type"], "error")
            self.assertIn("确认", event["message"])
            sync.assert_not_called()
            stream.assert_not_called()

    def test_relevance_uses_only_chunks_that_fit_context_budget(self):
        self.config["generation"]["max_context_chars"] = 150
        oversized_header = Document(page_content="短正文", metadata={"source_file": "长文件名" * 100})
        context = prepare_rag_context("问题", [(oversized_header, 0.9), (self.doc, 0.01)])
        self.assertEqual(context["generation_mode"], "low")
        self.assertEqual(context["top_score"], 0.01)
        self.assertEqual(context["sources"], ["a.pdf"])

    def test_local_api_bypasses_environment_proxy_in_both_modes(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            """真实本机 HTTP 服务模拟合法响应，不调用或测量模型。"""
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append(payload)
                result = {"model": "test", "done": True, "done_reason": "stop", "message": {"content": "概念说明。"}}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(result).encode() + b"\n")

            def log_message(self, *args):
                pass  # 测试不输出常规访问日志。

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.config["llm"]["base_url"] = f"http://127.0.0.1:{server.server_port}"
        context = prepare_rag_context("概念？", [])
        try:
            # 不可用的本机代理若被误用，将无法取得服务响应。
            with patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
                                         "no_proxy": "", "NO_PROXY": ""}):
                self.assertEqual(generate_answer("概念？", context)["generation_mode"], "empty")
                self.assertEqual(list(stream_answer("概念？", context))[-1]["type"], "done")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual([request["stream"] for request in received], [False, True])


if __name__ == "__main__":
    unittest.main()
