"""5.2.3 缓存与降级策略：TestDegradationContext。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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


if __name__ == "__main__":
    unittest.main()
