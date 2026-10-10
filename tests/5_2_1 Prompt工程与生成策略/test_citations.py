"""5.2.1 Prompt工程与生成策略：TestCitations。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from src.generation.rag_pipeline import build_context, resolve_citations


class TestCitations(unittest.TestCase):
    """校验引用编号和原文映射，不把映射通过等同于语义支持。"""

    def setUp(self):
        from src.utils.config import load_config
        self.config = load_config()
        config_patch = patch("src.generation.rag_pipeline.load_config", return_value=self.config)
        config_patch.start()
        self.addCleanup(config_patch.stop)
        self.documents = [Document(page_content=f"证据{i}：编码器的结构。", metadata={
            "source_file": "attention.pdf", "page_number": i + 2,
            "doc_id": "paper-attention", "chunk_id": f"chunk-{i}"}) for i in (1, 2, 3)]
        self.results = [(document, 1 - i / 10) for i, document in enumerate(self.documents)]
        self.context = build_context("编码器是什么结构？", self.results)


    def test_inline_pdf_location_and_authoritative_footer(self):
        answer = "## 回答\n结构如原文所述[参考文档1]。\n\n## 参考来源\n伪造.pdf，第999页"
        resolved = resolve_citations(answer, self.context)
        self.assertIn("结构如原文所述[参考文档1：attention.pdf；第3页（物理页码）]。", resolved["answer"])
        self.assertNotIn("伪造.pdf", resolved["answer"])
        self.assertNotIn("999", resolved["answer"])
        self.assertEqual(resolved["answer"].count("## 参考来源"), 1)
        self.assertEqual(resolved["citations"][0]["text"], self.documents[0].page_content)
        self.assertEqual(resolved["citations"][0]["metadata"]["chunk_id"], "chunk-1")
        self.assertEqual(resolved["warnings"], [])


    def test_word_and_text_do_not_get_invented_pdf_pages(self):
        for metadata, position in (({"source_file": "word.docx", "paragraph_index": 2}, "段落2"),
                                   ({"source_file": "word.docx", "table_index": 1}, "表格1"),
                                   ({"source_file": "text.md", "line_start": 2, "line_end": 6}, "行2–6")):
            with self.subTest(metadata=metadata):
                context = build_context("问题", [(Document(page_content="证据", metadata=metadata), 0.8)])
                resolved = resolve_citations("结论[参考文档1]。", context)
                self.assertIn(position, resolved["answer"])
                self.assertNotIn("物理页码", resolved["answer"])
                self.assertEqual(resolved["warnings"], [])


    def test_unknown_ids_are_marked_and_never_assigned_sources(self):
        resolved = resolve_citations("[参考文档0] [参考文档99] [参考文档99] [参考文档1]", self.context)
        self.assertEqual(resolved["invalid_citation_ids"], [0, 99])
        self.assertIn("[无效引用：参考文档99]", resolved["answer"])
        self.assertEqual([ref["id"] for ref in resolved["citations"]], [1])
        self.assertEqual(len(resolved["warnings"]), 2)


if __name__ == "__main__":
    unittest.main()
