"""5.1.2 文本分块策略：TestChunking。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import unittest
from langchain_core.documents import Document
from src.chunking.fixed_chunk import split_fixed
from src.chunking.recursive_chunk import split_recursive
from src.chunking.semantic_chunk import split_semantic


class TestChunking(unittest.TestCase):
    """用真实切分器检查内容、边界与溯源，不依赖模型和检索库。"""

    strategies = (split_fixed, split_recursive, split_semantic)


    def test_fixed_exact_overlap(self):
        """固定窗口按字符数切分，相邻正文块精确重叠。"""
        text = "甲乙丙丁戊己庚辛壬癸"
        chunks = split_fixed([Document(page_content=text)], 4, 1)
        self.assertEqual([chunk.page_content for chunk in chunks], ["甲乙丙丁", "丁戊己庚", "庚辛壬癸"])
        self.assertEqual([chunk.metadata["start_index"] for chunk in chunks], [0, 3, 6])


    def test_recursive_paragraph_boundaries(self):
        """优先按段落分隔符切分，并保留分隔符原文。"""
        text = "甲段内容。\n\n乙段内容。\n\n丙段内容。"
        chunks = split_recursive([Document(page_content=text)], 9, 0)
        self.assertEqual([chunk.page_content for chunk in chunks],
                         ["甲段内容。", "\n\n乙段内容。", "\n\n丙段内容。"])
        self.assertEqual("".join(chunk.page_content for chunk in chunks), text)


    def test_semantic_keeps_paragraphs(self):
        """短段落整体保留，空行跟随原段落。"""
        text = "甲段内容。\r\n\r\n乙段内容。\r\n\r\n丙段内容。"
        chunks = split_semantic([Document(page_content=text)], 10, 0)
        self.assertEqual([chunk.page_content for chunk in chunks],
                         ["甲段内容。\r\n\r\n", "乙段内容。\r\n\r\n", "丙段内容。"])


    def test_ranges_cover_original_content(self):
        """不同大小和重叠下，区间准确、向前推进，所有非空白字符均被覆盖。"""
        texts = ("重复段落。\n\n" * 30, "ABCD" * 50, "甲。乙！丙？\r\n\r\n" * 20)
        for splitter in self.strategies:
            for text in texts:
                for size, overlap in ((1, 0), (8, 0), (8, 2), (16, 15), (64, 10)):
                    with self.subTest(strategy=splitter.__name__, size=size, overlap=overlap, text=text[:12]):
                        chunks = splitter([Document(page_content=text)], size, overlap)
                        covered = set()
                        previous_start = -1
                        for chunk in chunks:
                            start, end = chunk.metadata["start_index"], chunk.metadata["end_index"]
                            self.assertGreater(start, previous_start)
                            self.assertLessEqual(len(chunk.page_content), size)
                            self.assertEqual(chunk.page_content, text[start:end])
                            covered.update(range(start, end))
                            previous_start = start
                        self.assertTrue(all(i in covered for i, char in enumerate(text) if not char.isspace()))


    def test_stable_distinct_chunk_ids(self):
        """重复执行 ID 不变，重复正文在不同页/段落及不同配置中不会混淆。"""
        documents = [Document(page_content="重复内容" * 8, metadata={"doc_id": "id", "page": page})
                     for page in (0, 1)]
        for splitter in self.strategies:
            chunks = splitter(documents, 12, 2)
            ids = [chunk.metadata["chunk_id"] for chunk in chunks]
            self.assertEqual(ids, [chunk.metadata["chunk_id"] for chunk in splitter(documents, 12, 2)])
            self.assertEqual(len(set(ids)), len(ids))
            changed = {chunk.metadata["chunk_id"] for chunk in splitter(documents, 16, 2)}
            self.assertTrue(set(ids).isdisjoint(changed))
        word = [Document(page_content="重复段落", metadata={"doc_id": "word", "block_index": index})
                for index in (1, 2)]
        self.assertEqual(len({chunk.metadata["chunk_id"] for chunk in split_fixed(word, 8)}), 2)


    def test_pdf_and_word_tables_remain_whole(self):
        """独立表格不拆，即使超长也保留跨页行信息和保护标记。"""
        rows = '[{"cells":["数据"],"page_number":4},{"cells":["结果"],"page_number":5}]'
        tables = [Document(page_content="| 表头 |\n| --- |\n| 数据 |\n| 结果 |", metadata={
            "doc_id": "id", "content_type": "table", "table_id": "id:table:1",
            "page": 3, "page_number": 4, "page_end": 5, "table_rows": rows,
        }), Document(page_content="表头\t值\n数据\t结果", metadata={"block_type": "table", "block_index": 2})]
        for splitter in self.strategies:
            chunks = splitter(tables, 4, 1)
            self.assertEqual([chunk.page_content for chunk in chunks], [table.page_content for table in tables])
            self.assertTrue(all(chunk.metadata["chunk_preserved"] == "table" for chunk in chunks))
            self.assertEqual(chunks[0].metadata["table_rows"], rows)
            self.assertEqual(chunks[0].metadata["page_end"], 5)


if __name__ == "__main__":
    unittest.main()
