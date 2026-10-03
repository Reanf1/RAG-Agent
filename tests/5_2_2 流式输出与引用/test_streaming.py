"""5.2.2 流式输出与引用：TestStreaming。"""

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
from src.generation.rag_pipeline import build_context, resolve_citations
from src.generation.streaming import stream_answer
from tests.helpers import StreamingResponse


class TestStreaming(unittest.TestCase):
    """验证真实迭代顺序、跨片段引用和异常；不把 mock 耗时当本地性能。"""

    def setUp(self):
        from src.utils.config import load_config
        self.config = load_config()
        patcher = patch("src.generation.rag_pipeline.load_config", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.context = build_context("层数？", [(Document(page_content="编码器有6层。", metadata={
            "source_file": "attention.pdf", "page_number": 3, "chunk_id": "功能样例块"}), 1.0)])
        self.done = {"model": "qwen2.5:7b", "done": True, "done_reason": "stop",
                     "message": {"content": ""}, "eval_count": 20, "prompt_eval_count": 400,
                     "total_duration": 1000000000}
        patcher = patch("src.generation.streaming.urlopen")
        self.opener = patcher.start()
        self.addCleanup(patcher.stop)

    def reply(self, chunks, *, done=True):
        packets = [{"message": {"content": chunk}, "done": False} for chunk in chunks]
        if done:
            packets.append(self.done)
        response = StreamingResponse(packets)
        self.opener.return_value = response
        return response

    def test_first_token_precedes_later_packets_and_uses_streaming_config(self):
        response = self.reply(["编码器有6层。", "[参考文档1]"])
        events = stream_answer("层数？", self.context, options={"seed": 17})
        first = next(events)
        self.assertEqual(first["type"], "token")
        self.assertEqual(first["answer"], "编码器有6层。")
        self.assertEqual(response.read_packets, 1)
        self.assertFalse(response.closed)
        payload = json.loads(self.opener.call_args.args[0].data)
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["options"]["seed"], 17)
        self.assertEqual(payload["options"]["top_k"], self.config["llm"]["top_k"])
        self.assertIn("第3页", next(events)["answer"])
        final = next(events)
        self.assertEqual(final["type"], "done")
        self.assertEqual(final["usage"]["eval_count"], 20)
        self.assertIn("## 参考来源", final["answer"])
        self.assertEqual(list(events), [])
        self.assertTrue(response.closed)

    def test_every_citation_split_updates_exact_position_before_done(self):
        marker = "[参考文档1]"
        for index in range(1, len(marker)):
            with self.subTest(index=index):
                self.reply(["事实" + marker[:index], marker[index:], "。更多文字"])
                events = list(stream_answer("层数？", self.context))
                self.assertEqual(events[0]["answer"], "事实")
                self.assertEqual(events[0]["citations"], [])
                self.assertEqual(events[1]["answer"], "事实[参考文档1：attention.pdf；第3页（物理页码）]")
                self.assertEqual(events[1]["citations"][0]["text"], "编码器有6层。")
                self.assertEqual(events[-1]["raw_answer"], "事实" + marker + "。更多文字")

    def test_single_character_chunks_keep_code_escape_and_hide_fake_link(self):
        raw = ("代码 `[参考文档99]`；转义 \\[参考文档98]。\n"
               "```text\n[参考文档97]\n```\n事实[参考文档1](https://example.invalid/fake.pdf)。")
        self.reply(list(raw))
        events = list(stream_answer("层数？", self.context))
        for event in events:
            self.assertEqual(event.get("invalid_citation_ids"), [])
            self.assertNotIn("example.invalid", event["answer"])
        self.assertEqual([ref["id"] for ref in events[-1]["citations"]], [1])
        self.assertIn("代码 `[参考文档99]`", events[-1]["answer"])
        self.assertIn("```text\n[参考文档97]\n```", events[-1]["answer"])

    def test_english_footer_is_not_counted_as_body_evidence(self):
        for heading in ("## 参考来源", "## References", "## Reference Sources", "## Sources"):
            raw = "## Answer\nNo body citation.\n" + heading + "\n[参考文档1] 伪造来源.pdf"
            with self.subTest(heading=heading):
                self.reply(list(raw))
                events = list(stream_answer("层数？", self.context))
                self.assertTrue(all(not event["citations"] for event in events))
                self.assertNotIn("伪造来源", events[-1]["answer"])
                self.assertTrue(events[-1]["missing_citations"])
                self.assertTrue(resolve_citations(raw, self.context)["missing_citations"])

    def test_complete_footer_heading_does_not_warn_about_broken_tail(self):
        self.reply(["事实[参考文档1]\n## 参考来源"])
        result = list(stream_answer("层数？", self.context))[-1]
        self.assertEqual(result["type"], "done")
        self.assertEqual(result["warnings"], [])

    def test_unknown_number_is_flagged_at_closing_bracket(self):
        self.reply(["事实[参考文档9", "]"])
        events = list(stream_answer("层数？", self.context))
        self.assertEqual(events[0]["invalid_citation_ids"], [])
        self.assertEqual(events[1]["invalid_citation_ids"], [9])
        self.assertIn("无效引用", events[1]["answer"])
        self.assertEqual(events[-1]["citations"], [])

    def test_unfinished_tail_and_length_limit_both_warn(self):
        self.done["done_reason"] = "length"
        self.reply(["事实[参考文档1]。尾部[参考文档"])
        final = list(stream_answer("层数？", self.context))[-1]
        self.assertEqual(final["type"], "done")
        self.assertTrue(final["answer"].startswith("事实[参考文档1：attention.pdf；第3页（物理页码）]。尾部\n"))
        self.assertEqual(len(final["warnings"]), 2)
        self.assertIn("Token 上限", final["warnings"][0])
        self.assertIn("未完成", final["warnings"][1])
        self.assertTrue(final["raw_answer"].endswith("[参考文档"))

    def test_eof_without_done_keeps_partial_text_and_never_returns_done(self):
        response = self.reply(["部分事实[参考文档1]。"], done=False)
        events = list(stream_answer("层数？", self.context))
        self.assertEqual([event["type"] for event in events], ["token", "error"])
        self.assertIn("未收到完成标记", events[-1]["message"])
        self.assertIn("部分事实", events[-1]["answer"])
        self.assertNotIn("usage", events[-1])
        self.assertTrue(response.closed)

    def test_malformed_ndjson_and_server_errors_preserve_prior_text(self):
        for tail in (b"bad-json\n", b'{"error":"model unavailable"}\n'):
            self.opener.return_value = BytesIO(b'{"message":{"content":"partial"}}\n' + tail)
            events = list(stream_answer("层数？", self.context))
            self.assertEqual(events[-1]["type"], "error")
            self.assertEqual(events[-1]["answer"], "partial")
            self.assertTrue(self.opener.return_value.closed)

    def test_network_failure_is_not_retried_or_returned_as_answer(self):
        self.opener.side_effect = URLError("本机服务不可用")
        events = list(stream_answer("层数？", self.context))
        self.assertEqual(events[0]["type"], "error")
        self.assertEqual(events[0]["answer"], "")
        self.assertEqual(self.opener.call_count, 1)

    def test_blank_done_is_error_and_done_packet_content_is_not_lost(self):
        self.reply([])
        self.assertEqual(list(stream_answer("层数？", self.context))[-1]["type"], "error")
        self.done["message"] = {"content": "最后一包[参考文档1]"}
        self.reply(["开头"])
        result = list(stream_answer("层数？", self.context))[-1]
        self.assertEqual(result["raw_answer"], "开头最后一包[参考文档1]")

    def test_closing_consumer_closes_http_connection_without_reading_rest(self):
        response = self.reply(["开头", "后文"])
        events = stream_answer("层数？", self.context)
        next(events)
        events.close()
        self.assertTrue(response.closed)
        self.assertEqual(response.read_packets, 1)

    def test_invalid_request_returns_error_before_network(self):
        for question, options in ((" ", None), ("层数？", {"top_k": 0})):
            result = list(stream_answer(question, self.context, options=options))
            self.assertEqual([event["type"] for event in result], ["error"])
        self.config["llm"]["base_url"] = "https://example.com"
        self.assertEqual(list(stream_answer("层数？", self.context))[0]["type"], "error")
        self.opener.assert_not_called()

    def test_empty_context_and_context_snapshot_remain_explicit(self):
        before = deepcopy(self.context)
        self.reply(["事实[参考文档1]"])
        list(stream_answer("层数？", self.context))
        self.assertEqual(self.context, before)
        empty = build_context("问题", [])
        self.reply(["当前知识库中未找到相关文档。"])
        final = list(stream_answer("问题", empty))[-1]
        self.assertEqual(final["warnings"], [])
        self.assertFalse(final["missing_citations"])
        self.assertIn("当前知识库中未找到相关文档。", json.loads(self.opener.call_args.args[0].data)["messages"][1]["content"])

    def test_error_categories_provide_retry_advice_without_retry_or_usage(self):
        for failure, expected in ((URLError("refused"), "无法连接"),
                                  (TimeoutError("timed out"), "超时"),
                                  (URLError(TimeoutError("timed out")), "超时"),
                                  (HTTPError("http://localhost", 404, "missing", {}, BytesIO(b"missing model")), "404")):
            self.opener.reset_mock()
            self.opener.side_effect = failure
            event = list(stream_answer("层数？", self.context))[-1]
            self.assertIn(expected, event["message"])
            self.assertIn("重新提交问题", event["retry_advice"])
            self.assertIn(type(failure).__name__, event["error_detail"])
            self.assertNotIn("usage", event)
            self.assertEqual(self.opener.call_count, 1)

    def test_bad_response_shape_and_service_error_keep_partial_answer(self):
        for tail, expected in (([], "格式"), ({"message": {"content": 123}}, "格式"),
                               ({"done": True}, "格式"),
                               ({"error": "out of memory"}, "生成失败")):
            self.opener.return_value = StreamingResponse([{"message": {"content": "已有部分事实。"}}, tail])
            event = list(stream_answer("层数？", self.context))[-1]
            self.assertEqual(event["type"], "error")
            self.assertIn(expected, event["message"])
            self.assertEqual(event["answer"], "已有部分事实。")
            self.assertIn("重新提交问题", event["retry_advice"])

    def test_empty_notice_is_visible_during_stream_and_in_final_answer(self):
        context = build_context("概念？", [])
        self.reply(["这是", "概念解释。"])
        events = list(stream_answer("概念？", context))
        self.assertTrue(all(event["answer"].startswith("当前知识库中未找到相关文档。") for event in events))
        self.assertTrue(all(event["citations"] == [] for event in events))
        self.assertEqual(events[-1]["generation_mode"], "empty")


if __name__ == "__main__":
    unittest.main()
