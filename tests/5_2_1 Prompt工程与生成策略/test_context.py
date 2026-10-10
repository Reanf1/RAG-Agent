"""5.2.1 Prompt工程与生成策略：TestContextBuilding。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from src.generation.prompt_template import build_rag_messages
from src.generation.rag_pipeline import build_context


class TestContextBuilding(unittest.TestCase):
    """按明确分数与字符预算核验拼接，不把字符数当作模型 Token 数。"""

    def setUp(self):
        from src.utils.config import load_config
        self.config = load_config()
        config_patch = patch("src.generation.rag_pipeline.load_config", return_value=self.config)
        config_patch.start()
        self.addCleanup(config_patch.stop)


    def test_full_text_and_sources_are_preserved(self):
        text = "  $x_{i}={a}$\r\n```json\n{\"k\": 5}\n```\n中英 AI 😀\n"
        results = [(Document(page_content=text, metadata={"source_file": "数学.md"}), 0.8),
                   (Document(page_content="另一个原文块", metadata={"source_file": "数学.md"}), 0.7)]
        context = build_context("问题", results)
        self.assertIn(text, context["context"])
        self.assertEqual(context["sources"], ["数学.md"])
        self.assertEqual(context["chunk_count"], 2)
        self.assertIn("\n\n---\n\n[参考文档2] 来源: 数学.md", context["context"])
        self.assertEqual(context["context_chars"], len(context["context"]))
        self.assertEqual(context["prompt_chars"], sum(len(m.content) for m in
                         build_rag_messages("问题", context["context"])))


    def test_truncation_prefers_late_sentence_or_paragraph_boundary(self):
        for first, limit in (("开头" * 10 + "。", 26), ("An important conclusion.", 32),
                             ("第一段" + "内容" * 10 + "\n", 30)):
            with self.subTest(first=first):
                document = Document(page_content=first + " More内容" * 100,
                                    metadata={"source_file": "论文.pdf", "page_number": 3})
                full = build_context("问题", [(document, 0.9)])
                header = full["context"].split("\n", 1)[0] + "\n"
                self.config["generation"]["max_context_chars"] = len(header) + limit + len("\n[正文已截断]")
                result = build_context("问题", [(document, 0.9)])
                self.assertEqual(result["context"], header + first.rstrip() + "\n[正文已截断]")
                self.assertTrue(result["truncated"])
                self.assertLessEqual(result["context_chars"], result["context_budget_chars"])
                self.config["generation"]["max_context_chars"] = 6000


if __name__ == "__main__":
    unittest.main()
