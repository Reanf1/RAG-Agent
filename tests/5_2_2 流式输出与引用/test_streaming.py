"""5.2.2 流式输出与引用：TestStreaming。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import json
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from src.generation.rag_pipeline import build_context
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


    def test_eof_without_done_keeps_partial_text_and_never_returns_done(self):
        response = self.reply(["部分事实[参考文档1]。"], done=False)
        events = list(stream_answer("层数？", self.context))
        self.assertEqual([event["type"] for event in events], ["token", "error"])
        self.assertIn("未收到完成标记", events[-1]["message"])
        self.assertIn("部分事实", events[-1]["answer"])
        self.assertNotIn("usage", events[-1])
        self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()
