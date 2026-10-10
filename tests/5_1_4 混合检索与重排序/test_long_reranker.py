"""长块边界回归：模型替身仅验证完整窗口输入，不冒称真实重排质量。"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import unittest
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from src.retrieval.reranker import Reranker
from src.generation.rag_pipeline import build_context


class CharacterTokenizer:
    """一个字符对应一个Token，使窗口越界、末尾丢失都能精确断言。"""

    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def num_special_tokens_to_add(self, **kwargs):
        return 4

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class TestLongReranker(unittest.TestCase):
    def setUp(self):
        self.reranker = Reranker()
        self.reranker.max_length = 160
        self.model = Mock(tokenizer=CharacterTokenizer())
        self.model.predict.side_effect = lambda pairs, **kwargs: [0.9 if "末行11" in text or "末行99" in text else 0.1 for _, text in pairs]
        patcher = patch("src.retrieval.reranker.get_reranker", return_value=self.model)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_tail_row_scored_with_header_and_actual_excerpt_sent_to_prompt(self):
        """所有行进入窗口；末行最高分的摘录与引用正文一致，原索引文档不变。"""
        header = "Table 1\n| 模型 | 精度 |\n| --- | --- |\n"
        rows = [f"| 末行{i} | {i}.7 |\n" for i in range(12)]
        document = Document(page_content=header + "".join(rows), metadata={"content_type": "table", "chunk_id": "table", "source_file": "长表.pdf", "page_number": 3})
        found = self.reranker.rerank("末行11是多少？", [(document, 1)])
        pairs = self.model.predict.call_args.args[0]
        self.assertGreater(len(pairs), 1)
        for row in rows:
            self.assertTrue(any(row in text for _, text in pairs))
        self.assertTrue(all(text.startswith(header) for _, text in pairs))
        self.assertTrue(all(len(q) + len(text) + 4 <= 160 for q, text in pairs))
        self.assertNotIn("rerank_excerpt", document.metadata)
        context = build_context("末行11是多少？", found)
        self.assertIn("末行11", context["context"])
        self.assertEqual(context["references"][0]["text"], found[0][0].metadata["rerank_excerpt"])
        self.assertEqual(found[0][0].metadata["chunk_id"], "table")

    def test_long_body_keeps_tail_and_window_boundaries(self):
        document = Document(page_content="正文" * 400 + "末行99")
        found = self.reranker.rerank("末尾内容", [(document, 1)])
        pairs = self.model.predict.call_args.args[0]
        self.assertIn("末行99", found[0][0].metadata["rerank_excerpt"])
        self.assertTrue(all(len(q) + len(text) + 4 <= 160 for q, text in pairs))

    def test_short_body_rerank_does_not_reuse_previous_excerpt(self):
        """正文已聚焦后重新评分，生成输入须采用本次正文，原候选保持不变。"""
        document = Document(page_content="本次正文末行99", metadata={"rerank_excerpt": "上次无关摘录",
                            "retrieval_warning": "长块按模型Token窗口精排，本次引用只覆盖选中的摘录；完整内容见原页。"})
        found = self.reranker.rerank("当前内容", [(document, 1)])
        self.assertNotIn("rerank_excerpt", found[0][0].metadata)
        self.assertEqual(build_context("当前内容", found)["references"][0]["text"], document.page_content)
        self.assertEqual(document.metadata["rerank_excerpt"], "上次无关摘录")

    def test_oversized_single_row_is_explicit_error_before_inference(self):
        document = Document(page_content="| 字段 |\n| --- |\n| " + "长" * 200 + " |", metadata={"content_type": "table"})
        with self.assertRaisesRegex(ValueError, "单行和表头"):
            self.reranker.rerank("查询", [(document, 1)])
        self.model.predict.assert_not_called()

    def test_oversized_query_is_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "问题超过"):
            self.reranker.rerank("问" * 160, [(Document(page_content="正文"), 1)])
        self.model.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
