"""模块一测试：临时文档与真实临时 Chroma；大模型只在独立实验中运行。"""

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf
from docx import Document as WordDocument
from docx.opc.exceptions import PackageNotFoundError
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 直接运行测试文件时，按文件位置加入项目根目录，不依赖当前工作目录。
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader.pdf_loader import load_pdf
from src.data_loader.docx_loader import load_docx
from src.data_loader.text_loader import load_text
from src.data_loader import batch_import, create_import_tasks, load_document
from src.chunking import split_documents
from src.chunking.fixed_chunk import split_fixed
from src.chunking.recursive_chunk import split_recursive
from src.chunking.semantic_chunk import split_semantic
from src.retrieval.vector_store import VectorStore, get_embeddings


class SmallEmbeddings(Embeddings):
    """测试使用明确的二维向量，不用于实验效果或速度结论。"""

    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def _vector(self, text):
        if "农业" in text:
            return [0.0, 1.0]
        if "反向" in text:
            return [-1.0, 0.0]
        return [1.0, 0.0]

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return self._vector(text)


class TestVectorStore(unittest.TestCase):
    """验证选定库的实际写入、查重、余弦查询、持久化和删除。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.embeddings = SmallEmbeddings()
        self.store = VectorStore(self.directory.name, self.embeddings)
        self.chunks = [
            Document(page_content="神经网络论文", metadata={
                "chunk_id": "a1", "doc_id": "a", "source_file": "论文A.pdf",
                "page_number": 2, "start_index": 0, "end_index": 7,
                "formula_layout": '[{"text":"x²","bbox":[1,2,3,4]}]',
            }),
            Document(page_content="农业数据论文", metadata={
                "chunk_id": "b1", "doc_id": "b", "source_file": "论文B.pdf", "page_number": 1,
            }),
            Document(page_content="反向向量论文", metadata={
                "chunk_id": "a2", "doc_id": "a", "source_file": "论文A.pdf", "page_number": 3,
            }),
        ]

    def test_empty_store_and_empty_query(self):
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(self.store.search("神经网络"), [])
        self.store.add_chunks(self.chunks)
        self.assertEqual(self.store.search(" \n"), [])
        self.assertEqual(self.store.search("神经网络", doc_id="missing"), [])
        self.assertEqual(self.embeddings.query_calls, [])

    def test_text_and_source_roundtrip(self):
        self.assertEqual(self.store.add_chunks(self.chunks), 3)
        actual = {chunk.metadata["chunk_id"]: chunk for chunk in self.store.list_chunks()}
        self.assertEqual(actual, {chunk.metadata["chunk_id"]: chunk for chunk in self.chunks})
        self.assertEqual(len(self.store.list_chunks("a")), 2)
        self.assertEqual(self.store.list_chunks("missing"), [])

    def test_cosine_order_and_negative_score(self):
        self.store.add_chunks(self.chunks)
        found = self.store.search("神经网络", k=10)
        self.assertEqual([doc.metadata["chunk_id"] for doc, _ in found], ["a1", "b1", "a2"])
        for (_, score), expected in zip(found, (1, 0, -1)):
            self.assertAlmostEqual(score, expected)

    def test_document_filter_and_top_k(self):
        self.store.add_chunks(self.chunks)
        self.assertEqual(len(self.store.search("神经网络", k=1)), 1)
        found = self.store.search("神经网络", doc_id="b")
        self.assertEqual([doc.metadata["chunk_id"] for doc, _ in found], ["b1"])

    def test_duplicate_import_does_not_encode_again(self):
        self.assertEqual(self.store.add_chunks(self.chunks + self.chunks), 3)
        self.assertEqual(self.store.add_chunks(self.chunks), 0)
        self.assertEqual(self.store.add_chunks([]), 0)
        self.assertEqual(self.embeddings.document_calls, [[chunk.page_content for chunk in self.chunks]])
        self.assertEqual(self.store.count(), 3)

    def test_incremental_import_only_encodes_new_chunks(self):
        self.store.add_chunks(self.chunks[:1])
        self.assertEqual(self.store.add_chunks(self.chunks), 2)
        self.assertEqual(self.embeddings.document_calls, [["神经网络论文"], ["农业数据论文", "反向向量论文"]])
        self.assertEqual(self.store.count(), 3)

    def test_delete_document_and_repeat_delete(self):
        self.store.add_chunks(self.chunks)
        self.assertEqual(self.store.delete_document("a"), 2)
        self.assertEqual(self.store.delete_document("a"), 0)
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.store.list_chunks("a"), [])
        self.assertEqual([doc.metadata["doc_id"] for doc, _ in self.store.search("神经网络")], ["b"])

    def test_reopen_preserves_content_without_reembedding(self):
        self.store.add_chunks(self.chunks)
        other_embeddings = SmallEmbeddings()
        reopened = VectorStore(self.directory.name, other_embeddings)
        self.assertEqual(reopened.count(), 3)
        self.assertEqual(reopened.search("神经网络", k=1)[0][0], self.chunks[0])
        self.assertEqual(other_embeddings.document_calls, [])

    def test_changed_model_or_index_parameters_are_rejected(self):
        from src.utils.config import load_config

        for section, key, value in (("embedding", "revision", "different-version"),
                                    ("retrieval", "search_ef", 10)):
            config = load_config()
            config[section][key] = value
            with self.subTest(key=key), patch("src.retrieval.vector_store.load_config", return_value=config):
                with self.assertRaisesRegex(ValueError, "使用新索引目录重建"):
                    VectorStore(self.directory.name, SmallEmbeddings())

    def test_invalid_ids_are_rejected_before_any_write(self):
        for metadata in ({}, {"chunk_id": "x"}, {"chunk_id": "", "doc_id": "a"}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.store.add_chunks([self.chunks[0], Document(page_content="坏块", metadata=metadata)])
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(self.embeddings.document_calls, [])
        for value in (None, "", "  "):
            with self.assertRaises(ValueError):
                self.store.delete_document(value)
        for k in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                self.store.search("问题", k=k)

    def test_same_text_in_different_documents_keeps_both_sources(self):
        second = Document(page_content=self.chunks[0].page_content,
                          metadata={"chunk_id": "c1", "doc_id": "c", "source_file": "论文C.pdf"})
        self.assertEqual(self.store.add_chunks([self.chunks[0], second]), 2)
        self.assertEqual({doc.metadata["doc_id"] for doc, _ in self.store.search("神经网络")}, {"a", "c"})

    def test_real_text_loading_and_batches_over_500(self):
        path = Path(self.directory.name) / "真实样本.md"
        path.write_text("# 神经网络\n\n论文使用农业数据。", encoding="utf-8")
        chunks = split_documents(load_document(path))
        self.assertEqual(self.store.add_chunks(chunks), len(chunks))
        self.assertEqual(self.store.search("农业")[0][0].metadata["source_file"], path.name)
        extra = [Document(page_content="神经网络", metadata={"chunk_id": f"extra-{i}", "doc_id": "extra"})
                 for i in range(501)]
        self.assertEqual(self.store.add_chunks(extra), 501)
        self.assertEqual(self.store.add_chunks(extra), 0)
        self.assertEqual(self.store.count(), 501 + len(chunks))


class TestLocalEmbeddings(unittest.TestCase):
    """隔离大模型，验证配置、本地约束和延迟加载缓存。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        get_embeddings.cache_clear()
        self.addCleanup(get_embeddings.cache_clear)
        self.config = {"embedding": {
            "local_path": self.directory.name, "device": "cpu", "batch_size": 32,
        }}
        patcher = patch("src.retrieval.vector_store.load_config", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_local_loading_and_normalization_arguments(self):
        """只从本地目录加载，关闭远程代码，文档编码开启归一化。"""
        with patch("src.retrieval.vector_store.HuggingFaceEmbeddings") as model:
            self.assertIs(get_embeddings(), model.return_value)
            model.assert_called_once_with(
                model_name=self.directory.name,
                model_kwargs={"device": "cpu", "local_files_only": True, "trust_remote_code": False},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )

    def test_repeated_calls_reuse_model(self):
        """多次调用复用已加载模型，不在每次提问时加载权重。"""
        with patch("src.retrieval.vector_store.HuggingFaceEmbeddings") as model:
            first = get_embeddings()
            self.assertIs(get_embeddings(), first)
            model.assert_called_once()

    def test_missing_model_does_not_initialize_remote_client(self):
        """缺少权重明确报错，不构造模型客户端或回退在线模式。"""
        self.config["embedding"]["local_path"] = str(Path(self.directory.name) / "missing")
        with patch("src.retrieval.vector_store.HuggingFaceEmbeddings") as model:
            with self.assertRaisesRegex(FileNotFoundError, "请先按用户手册下载权重"):
                get_embeddings()
            model.assert_not_called()


class TestEmbeddingEvaluation(unittest.TestCase):
    """用已知排名验证指标公式，模型效果来自独立的真实实验。"""

    def test_hit_recall_and_mrr_cutoffs(self):
        """排名六的相关项不算 Hit@5，排名十一的项不算 MRR@10。"""
        import numpy as np
        from reports.compare_embeddings import evaluate_rankings

        sample = {"corpus": [{"id": str(index)} for index in range(11)], "queries": [
            {"id": "q1", "relevant_ids": ["0"]},
            {"id": "q2", "relevant_ids": ["5", "10"]},
        ]}
        scores = np.asarray([list(range(11, 0, -1))] * 2)
        result = evaluate_rankings(scores, sample)
        self.assertEqual(result["hit_at_5"], 0.5)
        self.assertEqual(result["recall_at_5"], 0.5)
        self.assertAlmostEqual(result["mrr_at_10"], (1 + 1 / 6) / 2)
        self.assertEqual(result["queries"][1]["top10_ids"], [str(index) for index in range(10)])

    def test_language_groups_use_equal_weight_macro_average(self):
        """问题数不等时宏平均仍等权，并保留表现较差的语言组。"""
        import numpy as np
        from reports.compare_embeddings import evaluate_rankings

        sample = {"corpus": [{"id": str(index)} for index in range(6)], "queries": [
            {"id": "zh1", "group": "zh->zh", "relevant_ids": ["0"]},
            {"id": "zh2", "group": "zh->zh", "relevant_ids": ["5"]},
            {"id": "en1", "group": "en->en", "relevant_ids": ["5"]},
        ]}
        result = evaluate_rankings(np.asarray([list(range(6, 0, -1))] * 3), sample)
        self.assertAlmostEqual(result["hit_at_5"], 1 / 3)
        self.assertEqual(result["groups"]["zh->zh"]["query_count"], 2)
        self.assertEqual(result["groups"]["zh->zh"]["hit_at_5"], 0.5)
        self.assertEqual(result["macro"]["hit_at_5"], 0.25)
        self.assertEqual(result["worst_group_hit_at_5"], 0)


class TestChunking(unittest.TestCase):
    """用真实切分器检查内容、边界与溯源，不依赖模型和检索库。"""

    strategies = (split_fixed, split_recursive, split_semantic)

    def test_short_document(self):
        """短文档保持原文，包括缩进与尾部换行。"""
        text = "  中文论文摘要。\r\n"
        for splitter in self.strategies:
            with self.subTest(strategy=splitter.__name__):
                chunks = splitter([Document(page_content=text)], 32, 4)
                self.assertEqual([chunk.page_content for chunk in chunks], [text])
                self.assertEqual(chunks[0].metadata["start_index"], 0)
                self.assertEqual(chunks[0].metadata["end_index"], len(text))

    def test_empty_documents(self):
        """空列表、空文本和仅空白文档不生成无效块。"""
        for splitter in self.strategies:
            self.assertEqual(splitter([], 16, 0), [])
            self.assertEqual(splitter([Document(page_content=text) for text in ("", " \n\t")], 16, 0), [])

    def test_invalid_parameters(self):
        """拒绝不合法大小和重叠，避免零步长或负数切片。"""
        for splitter in self.strategies:
            for size, overlap in ((0, 0), (-1, 0), (4, -1), (4, 4), (4, 5),
                                  (4.5, 0), (4, 1.5), (True, 0), (4, False)):
                with self.subTest(strategy=splitter.__name__, size=size, overlap=overlap):
                    with self.assertRaises(ValueError):
                        splitter([], size, overlap)

    def test_fixed_exact_overlap(self):
        """固定窗口按字符数切分，相邻正文块精确重叠。"""
        text = "甲乙丙丁戊己庚辛壬癸"
        chunks = split_fixed([Document(page_content=text)], 4, 1)
        self.assertEqual([chunk.page_content for chunk in chunks], ["甲乙丙丁", "丁戊己庚", "庚辛壬癸"])
        self.assertEqual([chunk.metadata["start_index"] for chunk in chunks], [0, 3, 6])

    def test_fixed_no_redundant_tail(self):
        """末尾已被覆盖时不再输出完全包含在上一块中的尾块。"""
        for text, expected in (("12345678", ["12345678"]), ("1234567890", ["12345678", "67890"])):
            chunks = split_fixed([Document(page_content=text)], 8, 3)
            self.assertEqual([chunk.page_content for chunk in chunks], expected)

    def test_fixed_course_sizes(self):
        """课程规定的 256/512/1024 都可切分，正文长度不超过设置。"""
        text = "科研文本" * 600
        for size in (256, 512, 1024):
            chunks = split_fixed([Document(page_content=text)], size, 64)
            self.assertGreater(len(chunks), 1)
            self.assertTrue(all(len(chunk.page_content) <= size for chunk in chunks))
            self.assertEqual(chunks[-1].metadata["end_index"], len(text))

    def test_recursive_paragraph_boundaries(self):
        """优先按段落分隔符切分，并保留分隔符原文。"""
        text = "甲段内容。\n\n乙段内容。\n\n丙段内容。"
        chunks = split_recursive([Document(page_content=text)], 9, 0)
        self.assertEqual([chunk.page_content for chunk in chunks],
                         ["甲段内容。", "\n\n乙段内容。", "\n\n丙段内容。"])
        self.assertEqual("".join(chunk.page_content for chunk in chunks), text)

    def test_recursive_character_fallback(self):
        """没有分隔符的长文本仍能限长并保留重叠。"""
        chunks = split_recursive([Document(page_content="ABCDEFGHIJ")], 4, 1)
        self.assertEqual([chunk.page_content for chunk in chunks], ["ABCD", "DEFG", "GHIJ"])

    def test_semantic_keeps_paragraphs(self):
        """短段落整体保留，空行跟随原段落。"""
        text = "甲段内容。\r\n\r\n乙段内容。\r\n\r\n丙段内容。"
        chunks = split_semantic([Document(page_content=text)], 10, 0)
        self.assertEqual([chunk.page_content for chunk in chunks],
                         ["甲段内容。\r\n\r\n", "乙段内容。\r\n\r\n", "丙段内容。"])

    def test_semantic_chinese_sentences_and_quotes(self):
        """长段落拆句，结束引号与标点跟随原句。"""
        text = "第一句。”第二句！第三句？第四句。"
        chunks = split_semantic([Document(page_content=text)], 8, 0)
        self.assertEqual([chunk.page_content for chunk in chunks], ["第一句。”", "第二句！第三句？", "第四句。"])

    def test_semantic_english_decimal_and_doi(self):
        """英文句号可拆句，小数和 DOI 内部的点不误拆。"""
        text = "Value 3.14. DOI 10.1000/xyz. Result good."
        chunks = split_semantic([Document(page_content=text)], 27, 0)
        self.assertEqual([chunk.page_content for chunk in chunks],
                         ["Value 3.14.", " DOI 10.1000/xyz.", " Result good."])

    def test_semantic_complete_unit_overlap(self):
        """重叠复用完整句子；不足容纳整句时不截出半句。"""
        document = Document(page_content="甲甲。乙乙。丙丙。丁丁。")
        chunks = split_semantic([document], 6, 3)
        self.assertEqual([chunk.page_content for chunk in chunks], ["甲甲。乙乙。", "乙乙。丙丙。", "丙丙。丁丁。"])
        chunks = split_semantic([document], 6, 2)
        self.assertEqual([chunk.page_content for chunk in chunks], ["甲甲。乙乙。", "丙丙。丁丁。"])

    def test_semantic_long_sentence_fallback(self):
        """超长单句按字符兜底，最后短片段不丢失。"""
        text = "没有任何句号的超长文本用于测试"
        chunks = split_semantic([Document(page_content=text)], 5, 2)
        self.assertEqual([chunk.page_content for chunk in chunks], [text[i:i + 5] for i in range(0, len(text), 5)])

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

    def test_preserves_metadata_without_mutating_input(self):
        """保留 PDF 页码、版面和公式原位置，并复制元数据。"""
        metadata = {"doc_id": "论文ID", "source_file": "论文.pdf", "page": 2, "page_number": 3,
                    "layout": "two_column", "formula_layout": '[{"text":"x²"}]'}
        document = Document(page_content="公式 x^{2}。\n实验结果。" * 5, metadata=metadata)
        for splitter in self.strategies:
            chunks = splitter([document], 20, 2)
            for chunk in chunks:
                for key, value in metadata.items():
                    self.assertEqual(chunk.metadata[key], value)
            chunks[0].metadata["source_file"] = "改名.pdf"
            self.assertEqual(document.metadata, metadata)
            self.assertNotIn("chunk_id", document.metadata)

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

    def test_loaded_text_line_numbers(self):
        """真实 TXT 加载后细化 CRLF 行范围；块内原文和文件来源保持一致。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "文本.txt"
            path.write_bytes("甲乙\r\n丙丁\r\n戊己".encode("utf-8"))
            documents = load_text(path)
            chunks = split_fixed(documents, 4, 0)
            self.assertEqual([(chunk.metadata["line_start"], chunk.metadata["line_end"]) for chunk in chunks],
                             [(1, 1), (2, 2), (3, 3)])
            self.assertTrue(all(chunk.metadata["source"] == str(path.resolve()) for chunk in chunks))
            self.assertEqual(documents[0].metadata["line_end"], 3)
            # 恰好从 CRLF 中间开始的块仍归原行，不提前增加行号。
            chunks = split_fixed(documents, 3, 0)
            self.assertEqual([(chunk.metadata["line_start"], chunk.metadata["line_end"]) for chunk in chunks],
                             [(1, 1), (1, 2), (2, 3), (3, 3)])

    def test_loaded_word_and_pdf(self):
        """真实 Word/PDF 加载结果可直接分块，来源位置与表格不丢失。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "论文.docx"
            word = WordDocument()
            word.add_paragraph("研究背景。实验方法。研究结论。" * 3)
            word.add_table(rows=1, cols=2).cell(0, 0).text = "完整表格"
            word.save(path)
            for splitter in self.strategies:
                chunks = splitter(load_docx(path), 10, 2)
                self.assertTrue(all("block_index" in chunk.metadata and "page" not in chunk.metadata for chunk in chunks))
                self.assertEqual(chunks[-1].page_content, "完整表格\t")
                self.assertEqual(chunks[-1].metadata["chunk_preserved"], "table")
            path = Path(directory) / "论文.pdf"
            with pymupdf.open() as pdf:
                page = pdf.new_page()
                page.insert_text((72, 72), "研究背景。实验方法。研究结论。", fontname="china-s")
                pdf.save(path)
            for splitter in self.strategies:
                chunks = splitter(load_pdf(path), 10, 2)
                self.assertTrue(all(chunk.metadata["page_number"] == 1 for chunk in chunks))

    def test_config_defaults_and_overrides(self):
        """统一入口读取唯一配置，显式参数可覆盖实验值。"""
        documents = [Document(page_content="1234567890")]
        with patch("src.chunking.load_config", return_value={
            "chunking": {"strategy": "fixed", "chunk_size": 4, "chunk_overlap": 1},
        }):
            self.assertEqual([chunk.page_content for chunk in split_documents(documents)], ["1234", "4567", "7890"])
            chunks = split_documents(documents, strategy="semantic", chunk_size=8, chunk_overlap=0)
            self.assertEqual([chunk.page_content for chunk in chunks], ["12345678", "90"])
            self.assertTrue(all(chunk.metadata["chunk_strategy"] == "semantic" for chunk in chunks))

    def test_unknown_strategy(self):
        """未知策略明确报错，不悄悄回退。"""
        with self.assertRaisesRegex(ValueError, "不支持的分块策略"):
            split_documents([], "unknown", 8, 0)


class TestPDFLoader(unittest.TestCase):
    """验证实际 PDF 解析、页码和常见导入失败，不用 mock 替代解析器。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "论文.PDF"

    def write_pdf(self, texts):
        """生成可重复的中文/英文测试页面，空字符串对应空白页。"""
        with pymupdf.open() as pdf:
            for text in texts:
                page = pdf.new_page()
                if text:
                    page.insert_text((72, 72), text, fontname="china-s")
            pdf.save(self.path)

    def test_chinese_text_and_page_metadata(self):
        """每页正文与来源完整，页码同时提供索引和展示值。"""
        self.write_pdf(["中文论文摘要", "实验结果与结论"])
        documents = load_pdf(self.path)
        self.assertEqual(len(documents), 2)
        self.assertEqual([d.page_content for d in documents], ["中文论文摘要", "实验结果与结论"])
        for index, document in enumerate(documents):
            self.assertIsInstance(document, Document)
            self.assertEqual(document.metadata, {
                "source": str(self.path.resolve()),
                "source_file": "论文.PDF",
                "file_type": ".pdf",
                "doc_id": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "page": index,
                "page_number": index + 1,
                "total_pages": 2,
            })

    def test_relative_string_path(self):
        """字符串相对路径也能加载，来源统一为绝对路径。"""
        self.write_pdf(["摘要"])
        documents = load_pdf(os.path.relpath(self.path))
        self.assertEqual(documents[0].metadata["source"], str(self.path.resolve()))

    def test_document_id_is_stable_and_shared(self):
        """重复导入与文件副本共用内容标识，各页不会随机生成不同 ID。"""
        self.write_pdf(["摘要", "结论"])
        copy_path = self.path.with_name("副本.pdf")
        copy_path.write_bytes(self.path.read_bytes())
        documents = load_pdf(self.path) + load_pdf(self.path) + load_pdf(copy_path)
        self.assertEqual(len({d.metadata["doc_id"] for d in documents}), 1)

    def test_document_id_changes_with_file_content(self):
        """文件内容改变后标识也改变，避免误用旧索引。"""
        self.write_pdf(["初始内容"])
        original_id = load_pdf(self.path)[0].metadata["doc_id"]
        self.path = self.path.with_name("更新.pdf")
        self.write_pdf(["更新内容"])
        self.assertNotEqual(load_pdf(self.path)[0].metadata["doc_id"], original_id)

    def test_position_sorting(self):
        """内容流先写下方再写上方时，正文仍按页面位置排序。"""
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 144), "BOTTOM")
            page.insert_text((72, 72), "TOP")
            pdf.save(self.path)
        text = load_pdf(self.path)[0].page_content
        self.assertLess(text.index("TOP"), text.index("BOTTOM"))

    def test_blank_page_does_not_shift_page_numbers(self):
        """跳过中间空白页仍保留原始物理页码，避免引用错页。"""
        self.write_pdf(["第一页", "", "第三页"])
        with self.assertLogs("src.data_loader.pdf_loader", level="WARNING") as logs:
            documents = load_pdf(self.path)
        self.assertEqual([d.metadata["page_number"] for d in documents], [1, 3])
        self.assertEqual([d.metadata["total_pages"] for d in documents], [3, 3])
        self.assertIn("第 2 页", logs.output[0])

    def test_image_only_pdf_is_not_silently_accepted(self):
        """将文字渲染为图片，验证没有文本层的扫描页不会伪装为导入成功。"""
        with pymupdf.open() as original:
            page = original.new_page()
            page.insert_text((72, 72), "Image-only paper")
            image = page.get_pixmap().tobytes("png")
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_image(page.rect, stream=image)
            pdf.save(self.path)
        with self.assertLogs("src.data_loader.pdf_loader", level="WARNING"):
            with self.assertRaisesRegex(ValueError, "未提取到文本"):
                load_pdf(self.path)

    def test_missing_file(self):
        """不存在的文件明确报错。"""
        with self.assertRaises(FileNotFoundError):
            load_pdf(self.path)

    def test_unsupported_extension(self):
        """其他加载器的格式不能误入 PDF 解析流程。"""
        path = self.path.with_suffix(".txt")
        path.write_text("摘要", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "不支持此文件格式"):
            load_pdf(path)

    def test_damaged_pdf(self):
        """损坏文件的解析错误向上传递，不返回空列表掩盖失败。"""
        self.path.write_bytes(b"not a valid PDF")
        with self.assertRaises(pymupdf.FileDataError):
            load_pdf(self.path)

    def test_password_protected_pdf(self):
        """需要打开密码的文件明确提示先解密。"""
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 72), "Protected paper")
            pdf.save(self.path, encryption=pymupdf.PDF_ENCRYPT_AES_256,
                     owner_pw="owner", user_pw="reader")
        with self.assertRaisesRegex(ValueError, "需要密码"):
            load_pdf(self.path)


class TestAcademicPDF(unittest.TestCase):
    """用实际绘制的论文版面验证阅读顺序、续表与公式，不 mock 解析器。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "academic.pdf"

    def draw_table(self, page, top, rows, caption="Table 1", three_lines=False):
        """绘制可重复的网格表或三线表，保留真正的文字和线段。"""
        if caption:
            page.insert_text((50, top - 15), caption)
        bottom = top + 30 * len(rows)
        columns = len(rows[0])
        if not three_lines:
            for index in range(columns + 1):
                x = 50 + 450 * index / columns
                page.draw_line((x, top), (x, bottom))
        ys = [top, top + 30, bottom] if three_lines else range(top, bottom + 1, 30)
        for y in ys:
            page.draw_line((50, y), (500, y))
        for row_index, row in enumerate(rows):
            for column, value in enumerate(row):
                page.insert_text((60 + 450 * column / columns, top + 20 + row_index * 30), value)

    def write_columns(self, page, top):
        """同一高度左右栏同时存在，朴素位置排序会交错读取。"""
        for index in range(3):
            for x, label in [(50, "LEFT"), (330, "RIGHT")]:
                page.insert_text((x, top + index * 25),
                                 f"{label}-{top}-{index} paragraph with many words", fontsize=9)

    def test_two_columns_with_full_width_heading(self):
        """先通栏标题，再完整左栏、完整右栏，最后页脚。"""
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            page.insert_text((190, 60), "ACADEMIC PAPER TITLE", fontsize=16)
            self.write_columns(page, 100)
            page.insert_text((270, 760), "FOOTER TEXT")
            pdf.save(self.path)
        text = load_pdf(self.path)[0].page_content
        self.assertLess(text.index("TITLE"), text.index("LEFT-100-0"))
        self.assertLess(text.index("LEFT-100-2"), text.index("RIGHT-100-0"))
        self.assertLess(text.index("RIGHT-100-2"), text.index("FOOTER"))

    def test_full_width_separator_between_column_sections(self):
        """页中通栏内容将双栏分为上下区域，避免跨区域串读。"""
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            self.write_columns(page, 100)
            page.insert_text((180, 220), "FULL WIDTH SECTION HEADING", fontsize=14)
            self.write_columns(page, 270)
            pdf.save(self.path)
        text = load_pdf(self.path)[0].page_content
        expected = ["LEFT-100-2", "RIGHT-100-0", "HEADING", "LEFT-270-0", "RIGHT-270-0"]
        self.assertEqual([text.index(token) for token in expected], sorted(text.index(token) for token in expected))

    def test_indented_single_column_is_not_reordered(self):
        """上下分离的缩进段落不能被误当作同时并排的两栏。"""
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            for y, x, label in [(100,330,"FIRST"),(125,330,"SECOND"),(300,50,"THIRD"),(325,50,"FOURTH")]:
                page.insert_text((x,y), label + " paragraph with enough words", fontsize=9)
            pdf.save(self.path)
        text = load_pdf(self.path)[0].page_content
        self.assertLess(text.index("FIRST"), text.index("THIRD"))

    def test_cross_page_table_merges_rows_and_keeps_row_sources(self):
        """相邻页同一续表合并，重复表头去重，每行记录真实来源页。"""
        with pymupdf.open() as pdf:
            self.draw_table(pdf.new_page(width=600,height=800), 650, [["Method","Score"],["A","90"],["B","95"]])
            self.draw_table(pdf.new_page(width=600,height=800), 70, [["Method","Score"],["C","96"],["D","97"]], "Table 1 (continued)")
            pdf.save(self.path)
        tables = [d for d in load_pdf(self.path) if d.metadata.get("content_type") == "table"]
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual(table.page_content.count("Method"), 1)
        self.assertIn("| C | 96 |", table.page_content)
        self.assertEqual((table.metadata["page_number"], table.metadata["page_end"]), (1,2))
        rows = json.loads(table.metadata["table_rows"])
        self.assertEqual([row["page_number"] for row in rows], [1,1,1,2,2])
        self.assertEqual(len(json.loads(table.metadata["table_regions"])), 2)

    def test_continuation_without_repeated_header(self):
        """明确同编号续表可无重复表头，下一页首行数据不可丢失。"""
        with pymupdf.open() as pdf:
            self.draw_table(pdf.new_page(width=600,height=800),650,[["Method","Score"],["A","90"],["B","91"]])
            self.draw_table(pdf.new_page(width=600,height=800),70,[["C","92"],["D","93"]],"Table 1 (continued)")
            pdf.save(self.path)
        tables = [d for d in load_pdf(self.path) if d.metadata.get("content_type") == "table"]
        self.assertEqual(len(tables), 1)
        self.assertIn("| C | 92 |", tables[0].page_content)
        self.assertIn("| D | 93 |", tables[0].page_content)

    def test_different_numbered_tables_do_not_merge(self):
        """几何位置和表头相同也不能合并不同编号的独立表。"""
        with pymupdf.open() as pdf:
            self.draw_table(pdf.new_page(width=600,height=800),650,[["Method","Score"],["A","90"]],"Table 1")
            self.draw_table(pdf.new_page(width=600,height=800),70,[["Method","Score"],["B","91"]],"Table 2")
            pdf.save(self.path)
        self.assertEqual(sum(d.metadata.get("content_type") == "table" for d in load_pdf(self.path)), 2)

    def test_nonadjacent_or_incompatible_tables_do_not_merge(self):
        """页码断开、表头或列结构不同、非页边续接均分别保留。"""
        for middle_page, top, rows in [(True,70,[["Method","Score"],["B","91"]]),
                                      (False,70,[["Data","Count"],["B","91"]]),
                                      (False,70,[["Method","Score","Time"],["B","91","2"]]),
                                      (False,350,[["Method","Score"],["B","91"]])]:
            with self.subTest(middle_page=middle_page,top=top,rows=rows):
                with pymupdf.open() as pdf:
                    self.draw_table(pdf.new_page(width=600,height=800),650,[["Method","Score"],["A","90"]])
                    if middle_page:
                        pdf.new_page(width=600,height=800).insert_text((50,100),"Intermediate page")
                    self.draw_table(pdf.new_page(width=600,height=800),top,rows,"Table 1")
                    pdf.save(self.path)
                self.assertEqual(sum(d.metadata.get("content_type") == "table" for d in load_pdf(self.path)), 2)

    def test_three_line_table_and_empty_cells(self):
        """三线表在规则围出的区域内识别，空单元格不从相邻行填值。"""
        with pymupdf.open() as pdf:
            self.draw_table(pdf.new_page(width=600,height=800),150,[["Method","Score"],["A","90"],["B",""],["C","95"]],three_lines=True)
            pdf.save(self.path)
        tables = [d for d in load_pdf(self.path) if d.metadata.get("content_type") == "table"]
        self.assertEqual(len(tables), 1)
        self.assertIn("| B |  |", tables[0].page_content)
        self.assertEqual(json.loads(tables[0].metadata["table_rows"])[2]["cells"], ["B", ""])

    def test_formula_symbols_and_scripts_are_preserved(self):
        """公式符号与上下标保持，不能把 x 的平方压成普通 x2。"""
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((60,80),"y = x",fontsize=14)
            page.insert_text((90,74),"2",fontsize=8)
            page.insert_text((100,80)," + a",fontsize=14)
            page.insert_text((124,85),"i",fontsize=8)
            page.insert_font(fontname="math",fontbuffer=pymupdf.Font("cjk").buffer)
            page.insert_text((60,120),"α + β = ∑ x / √n",fontname="math")
            pdf.save(self.path)
        document = load_pdf(self.path)[0]
        self.assertIn("x^{2}", document.page_content)
        self.assertIn("a_{i}", document.page_content)
        for symbol in ["α","β","∑","√"]:
            self.assertIn(symbol, document.page_content)
        layout = json.loads(document.metadata["formula_layout"])
        spans = [span for line in layout for span in line["spans"]]
        self.assertTrue(any(span["text"] == "2" and span["origin"][1] == 74 for span in spans))

    def test_multiple_table_styles_on_same_page(self):
        """同页网格表与三线表分别保留，不把横线范围或单元格混在一起。"""
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=600,height=800)
            self.draw_table(page,150,[["Method","Score"],["GRID","90"]],"Table 1")
            self.draw_table(page,400,[["Method","Score"],["PLAIN","95"],["OTHER","96"]],"Table 2",three_lines=True)
            pdf.save(self.path)
        tables=[d for d in load_pdf(self.path) if d.metadata.get("content_type")=="table"]
        self.assertEqual(len(tables),2)
        self.assertIn("GRID",tables[0].page_content)
        self.assertNotIn("PLAIN",tables[0].page_content)
        self.assertIn("PLAIN",tables[1].page_content)

    def test_three_line_formula_cells_and_header_words(self):
        """多词表头和公式中的间隔不拆成虚假的列。"""
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=600,height=800)
            self.draw_table(page,150,[["Layer Type","Complexity per Layer"],["Attention","O(n * d)"],["Recurrent","O(n * d * d)"]],three_lines=True)
            pdf.save(self.path)
        table=next(d for d in load_pdf(self.path) if d.metadata.get("content_type")=="table")
        self.assertEqual(len(json.loads(table.metadata["table_rows"])[0]["cells"]),2)
        self.assertIn("Complexity per Layer",table.page_content)
        self.assertIn("O(n * d * d)",table.page_content)

    def test_chinese_continuation_and_unnumbered_tables(self):
        """中文续表无重复表头时可合并；无编号的独立表不凭同表头合并。"""
        for numbered in [True,False]:
            with self.subTest(numbered=numbered):
                with pymupdf.open() as pdf:
                    first=pdf.new_page(width=600,height=800)
                    self.draw_table(first,650,[["Method","Score"],["A","90"]],"")
                    if numbered:
                        first.insert_text((50,635),"表 1",fontname="china-s")
                    second=pdf.new_page(width=600,height=800)
                    self.draw_table(second,70,[["B","91"],["C","92"]] if numbered else [["Method","Score"],["B","91"]],"")
                    if numbered:
                        second.insert_text((50,55),"续表 1",fontname="china-s")
                    pdf.save(self.path)
                tables=[d for d in load_pdf(self.path) if d.metadata.get("content_type")=="table"]
                self.assertEqual(len(tables),1 if numbered else 2)
                self.assertIn("B",tables[-1].page_content)

    def test_dense_formula_rows_do_not_leak_into_neighbors(self):
        """上标边缘与邻行相交时，不能向表头或邻行复制残留字符。"""
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=600,height=800)
            page.insert_text((50,85),"Table 1")
            for y in [100,120,158]:page.draw_line((50,y),(500,y))
            page.insert_text((60,115),"Method",fontsize=11)
            page.insert_text((300,115),"Complexity",fontsize=11)
            for y,label in [(130,"Attention"),(142,"Recurrent")]:
                page.insert_text((60,y),label,fontsize=11)
                page.insert_text((300,y),"x",fontsize=11)
                page.insert_text((306,y-4),"2",fontsize=7)
            pdf.save(self.path)
        table=next(d for d in load_pdf(self.path) if d.metadata.get("content_type")=="table")
        rows=json.loads(table.metadata["table_rows"])
        self.assertEqual([row["cells"] for row in rows],[["Method","Complexity"],["Attention","x^{2}"],["Recurrent","x^{2}"]])

    def test_ambiguous_three_line_header_keeps_original_text(self):
        """父表头横跨两个子列时不输出错误结构，正文和原文位置仍保留。"""
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=600,height=800)
            page.insert_text((50,85),"Table 1")
            for y in [100,145,220]:page.draw_line((50,y),(500,y))
            page.insert_text((60,120),"Method")
            page.insert_text((285,115),"Combined long parent heading")
            page.insert_text((285,138),"LEFT")
            page.insert_text((400,138),"RIGHT")
            for y,label in [(165,"A"),(190,"B")]:
                page.insert_text((60,y),label)
                page.insert_text((285,y),"90")
                page.insert_text((400,y),"95")
            pdf.save(self.path)
        with self.assertLogs("src.data_loader.pdf_loader",level="WARNING"):
            documents=load_pdf(self.path)
        self.assertFalse(any(d.metadata.get("content_type")=="table" for d in documents))
        self.assertIn("LEFT",documents[0].page_content)
        self.assertIn("RIGHT",documents[0].page_content)

    def test_three_page_table_keeps_first_caption_identity(self):
        """中间续页不重复标题时，仍保留整张表的编号和第三页数据。"""
        with pymupdf.open() as pdf:
            self.draw_table(pdf.new_page(width=600,height=800),650,[["Method","Score"],["A","90"],["B","91"]])
            self.draw_table(pdf.new_page(width=600,height=800),70,[["Method","Score"]]+[[f"M{i}",str(i)] for i in range(22)],"")
            self.draw_table(pdf.new_page(width=600,height=800),70,[["Method","Score"],["FINAL","99"]],"")
            pdf.save(self.path)
        tables=[d for d in load_pdf(self.path) if d.metadata.get("content_type")=="table"]
        self.assertEqual(len(tables),1)
        self.assertEqual(tables[0].metadata["page_end"],3)
        self.assertIn("FINAL",tables[0].page_content)

    def test_fraction_layout_retains_original_positions(self):
        """不猜测二维分式的 LaTeX，保留分子/分母位置以回看原文。"""
        with pymupdf.open() as pdf:
            page=pdf.new_page()
            page.insert_text((60,130),"f =",fontsize=12)
            page.insert_text((95,119),"1",fontsize=10)
            page.draw_line((92,125),(119,125))
            page.insert_text((95,140),"1 + x",fontsize=10)
            pdf.save(self.path)
        document = load_pdf(self.path)[0]
        spans = [s for line in json.loads(document.metadata["formula_layout"]) for s in line["spans"]]
        self.assertTrue(any(s["text"] == "1" and s["origin"][1] == 119 for s in spans))
        self.assertTrue(any("1 + x" in s["text"] and s["origin"][1] == 140 for s in spans))

    def test_image_formula_is_locatable_after_batch_save(self):
        """混合文本页的公式图片保留原文定位，批量保存后仍可裁剪查看。"""
        with pymupdf.open() as picture:
            image_page=picture.new_page(width=180,height=50)
            image_page.insert_text((10,30),"y = (a+b)/c",fontsize=16)
            image=image_page.get_pixmap().tobytes("png")
        with pymupdf.open() as pdf:
            page=pdf.new_page()
            page.insert_text((60,80),"Image equation follows:")
            page.insert_image(pymupdf.Rect(60,100,240,150),stream=image)
            pdf.save(self.path)
        tasks=create_import_tasks([("formula.pdf",self.path.read_bytes())])
        list(batch_import(tasks, Path(self.directory.name)/"raw"))
        self.assertEqual(tasks[0]["status"],"success")
        document=tasks[0]["documents"][0]
        region=json.loads(document.metadata["image_regions"])[0]
        self.assertIn("图像区域",document.page_content)
        with pymupdf.open(document.metadata["source"]) as saved:
            crop=saved[region["page_number"]-1].get_pixmap(clip=region["bbox"])
            self.assertEqual((crop.width,crop.height),(180,50))
            self.assertGreater(len(crop.tobytes("png")),100)


class TestDOCXLoader(unittest.TestCase):
    """通过实际 DOCX 验证正文、表格顺序和位置，不修改原始课程资料。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "论文.DOCX"

    def test_paragraph_table_order_and_metadata(self):
        """表格保留在正文原位置，中文段落与单元格内容均可读取。"""
        word = WordDocument()
        word.add_paragraph("摘要")
        table = word.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "方法"
        table.cell(0, 1).text = "准确率"
        table.cell(1, 0).text = "Transformer"
        table.cell(1, 1).text = "90%"
        word.add_paragraph("结论")
        word.save(self.path)

        documents = load_docx(os.path.relpath(self.path))
        self.assertEqual([d.page_content for d in documents], [
            "摘要", "方法\t准确率\nTransformer\t90%", "结论",
        ])
        self.assertEqual([d.metadata["block_index"] for d in documents], [1, 2, 3])
        self.assertEqual([d.metadata["block_type"] for d in documents], [
            "paragraph", "table", "paragraph",
        ])
        self.assertEqual(documents[0].metadata["paragraph_index"], 1)
        self.assertEqual(documents[1].metadata["table_index"], 1)
        self.assertEqual(documents[2].metadata["paragraph_index"], 2)
        for document in documents:
            self.assertIsInstance(document, Document)
            self.assertEqual(document.metadata["source"], str(self.path.resolve()))
            self.assertEqual(document.metadata["source_file"], "论文.DOCX")
            self.assertEqual(document.metadata["file_type"], ".docx")
            self.assertNotIn("page", document.metadata)
            self.assertNotIn("page_number", document.metadata)

    def test_blank_blocks_do_not_shift_positions(self):
        """空段落和空表格跳过，但后续段落/表格编号不偏移。"""
        word = WordDocument()
        word.add_paragraph("")
        word.add_table(rows=1, cols=1)
        word.add_paragraph("正文")
        word.add_table(rows=1, cols=1).cell(0, 0).text = "实验数据"
        word.save(self.path)
        documents = load_docx(self.path)
        self.assertEqual([d.metadata["block_index"] for d in documents], [3, 4])
        self.assertEqual(documents[0].metadata["paragraph_index"], 2)
        self.assertEqual(documents[1].metadata["table_index"], 2)

    def test_table_empty_cells_and_multiple_paragraphs(self):
        """空单元格保留分隔符，单元格内的多个段落不丢失。"""
        word = WordDocument()
        table = word.add_table(rows=1, cols=3)
        table.cell(0, 1).text = "第一段"
        table.cell(0, 1).add_paragraph("第二段")
        word.save(self.path)
        self.assertEqual(load_docx(self.path)[0].page_content, "\t第一段\n第二段\t")

    def test_document_id_is_stable_shared_and_changes(self):
        """同文件各块共用内容 ID，重复读取稳定，内容变化后更新。"""
        word = WordDocument()
        word.add_paragraph("摘要")
        word.add_paragraph("结论")
        word.save(self.path)
        documents = load_docx(self.path) + load_docx(self.path)
        ids = {d.metadata["doc_id"] for d in documents}
        self.assertEqual(ids, {hashlib.sha256(self.path.read_bytes()).hexdigest()})
        word.add_paragraph("新增实验")
        word.save(self.path)
        self.assertNotIn(load_docx(self.path)[0].metadata["doc_id"], ids)

    def test_empty_document(self):
        """仅有空段落和空表格时明确报错。"""
        word = WordDocument()
        word.add_paragraph(" \t ")
        word.add_table(rows=1, cols=2)
        word.save(self.path)
        with self.assertRaisesRegex(ValueError, "未提取到文本"):
            load_docx(self.path)

    def test_missing_file(self):
        """缺失文件不会被当成空文档。"""
        with self.assertRaises(FileNotFoundError):
            load_docx(self.path)

    def test_legacy_doc_extension(self):
        """旧版 .doc 格式需转换后导入。"""
        path = self.path.with_suffix(".doc")
        path.write_bytes(b"legacy document")
        with self.assertRaisesRegex(ValueError, "仅支持 .docx"):
            load_docx(path)

    def test_damaged_document(self):
        """损坏文件的解析错误向上传递。"""
        self.path.write_bytes(b"not a DOCX archive")
        with self.assertRaises(PackageNotFoundError):
            load_docx(self.path)


class TestTextLoader(unittest.TestCase):
    """验证纯文本保真、中文编码和行范围。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "文献.txt"

    def test_utf8_text_and_line_metadata(self):
        """中文正文保留首尾空白与空行，来源和行号完整。"""
        text = "\n  中文正文\n\n第二段\n"
        self.path.write_bytes(text.encode("utf-8"))
        documents = load_text(os.path.relpath(self.path))
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, text)
        self.assertEqual(documents[0].metadata, {
            "source": str(self.path.resolve()),
            "source_file": "文献.txt",
            "file_type": ".txt",
            "doc_id": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "line_start": 1,
            "line_end": 4,
        })

    def test_utf8_bom_and_crlf(self):
        """移除 BOM，但保留 Windows 的 CRLF 换行。"""
        text = "第一行\r\n第二行\r\n"
        self.path.write_bytes(text.encode("utf-8-sig"))
        document = load_text(self.path)[0]
        self.assertEqual(document.page_content, text)
        self.assertEqual(document.metadata["line_end"], 2)

    def test_markdown_preserves_syntax_and_indentation(self):
        """Markdown 标题、链接、表格和代码块不被转换或清理。"""
        text = "# 标题\n\n[论文](https://example.com)\n|方法|结果|\n|---|---|\n```python\n    print('中文')\n```\n"
        for suffix in [".MD", ".markdown"]:
            with self.subTest(suffix=suffix):
                path = self.path.with_suffix(suffix)
                path.write_bytes(text.encode("utf-8"))
                document = load_text(path)[0]
                self.assertEqual(document.page_content, text)
                self.assertEqual(document.metadata["file_type"], suffix.lower())
                self.assertNotIn("page_number", document.metadata)

    def test_document_id_is_stable_and_changes(self):
        """内容指纹由原始字节生成，重复读取稳定，文本变化后更新。"""
        self.path.write_text("原始正文", encoding="utf-8")
        original_id = load_text(self.path)[0].metadata["doc_id"]
        self.assertEqual(load_text(self.path)[0].metadata["doc_id"], original_id)
        self.path.write_text("更新正文", encoding="utf-8")
        self.assertNotEqual(load_text(self.path)[0].metadata["doc_id"], original_id)

    def test_empty_or_whitespace_only_file(self):
        """空字节、仅 BOM 和纯空白都不视为有效内容。"""
        for data in [b"", b"\xef\xbb\xbf", b" \t\r\n"]:
            with self.subTest(data=data):
                self.path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, "没有有效内容"):
                    load_text(self.path)

    def test_invalid_utf8(self):
        """不静默替换错误字符，也不猜测 GBK 等其他编码。"""
        self.path.write_bytes("中文".encode("gbk"))
        with self.assertRaises(UnicodeDecodeError):
            load_text(self.path)

    def test_missing_file(self):
        """缺失文件明确报错。"""
        with self.assertRaises(FileNotFoundError):
            load_text(self.path)

    def test_unsupported_extension(self):
        """二进制格式不能被当作纯文本导入。"""
        path = self.path.with_suffix(".pdf")
        path.write_bytes(b"PDF")
        with self.assertRaisesRegex(ValueError, "不支持此文件格式"):
            load_text(path)


class TestBatchImport(unittest.TestCase):
    """验证批量任务的独立状态、真实保存和有界重试。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.raw_dir = Path(self.directory.name) / "raw"

    def test_all_formats_saved_with_correct_sources(self):
        """四类文档均实际解析和保存，来源指向持久文件而非临时文件。"""
        word = WordDocument()
        word.add_paragraph("Word 摘要")
        buffer = io.BytesIO()
        word.save(buffer)
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((72, 72), "PDF abstract")
            pdf_data = pdf.tobytes()
        files = [("paper.pdf", pdf_data), ("paper.docx", buffer.getvalue()),
                 ("说明.txt", "文本摘要".encode("utf-8")), ("说明.md", b"# Abstract")]
        tasks = create_import_tasks(files)
        progress = list(batch_import(tasks, self.raw_dir))
        self.assertEqual(progress[-1], {"completed": 4, "total": 4})
        self.assertEqual([t["status"] for t in tasks], ["success"] * 4)
        for task in tasks:
            path = Path(task["path"])
            self.assertEqual(path.read_bytes(), task["data"])
            self.assertEqual(task["attempts"], 1)
            for document in task["documents"]:
                self.assertEqual(document.metadata["source"], str(path))
                self.assertEqual(document.metadata["source_file"], task["name"])

    def test_failure_does_not_stop_later_files_or_leave_failed_file(self):
        """中间文件损坏也继续导入后续文件，完成进度包含失败项。"""
        tasks = create_import_tasks([("first.txt", b"First"), ("bad.pdf", b"broken"),
                                     ("last.md", b"# Last")])
        events = batch_import(tasks, self.raw_dir)
        next(events)
        self.assertEqual(tasks[0]["status"], "pending")
        next(events)
        self.assertEqual(tasks[0]["status"], "loading")
        progress = list(events)
        self.assertEqual([t["status"] for t in tasks], ["success", "failed", "success"])
        self.assertIn("FileDataError", tasks[1]["error"])
        self.assertEqual(tasks[1]["documents"], [])
        self.assertEqual(tasks[1]["path"], "")
        self.assertEqual(progress[-1], {"completed": 3, "total": 3})
        self.assertEqual(len(list(self.raw_dir.rglob("*.*"))), 2)

    def test_retry_failed_recovers_without_reloading_success(self):
        """模拟一次临时错误，手动重试仅失败项，成功项尝试次数不增加。"""
        tasks = create_import_tasks([("good.md", b"Good"), ("retry.txt", b"Retry")])

        def fail_once(path):
            if path.name == "retry.txt":
                raise OSError("临时读取失败")
            return load_document(path)

        with patch("src.data_loader.load_document", side_effect=fail_once):
            list(batch_import(tasks, self.raw_dir))
        self.assertEqual([t["status"] for t in tasks], ["success", "failed"])
        progress = list(batch_import(tasks, self.raw_dir, retry_failed=True))
        self.assertEqual(progress[-1], {"completed": 1, "total": 1})
        self.assertEqual([t["status"] for t in tasks], ["success", "success"])
        self.assertEqual([t["attempts"] for t in tasks], [1, 2])
        self.assertEqual(tasks[1]["error"], "")

    def test_persistent_failure_is_bounded(self):
        """每次重试只处理一轮，永久错误不会产生无限循环。"""
        tasks = create_import_tasks([("bad.txt", b"\xff")])
        list(batch_import(tasks, self.raw_dir))
        list(batch_import(tasks, self.raw_dir, retry_failed=True))
        self.assertEqual(tasks[0]["attempts"], 2)
        self.assertEqual(tasks[0]["status"], "failed")
        self.assertIn("UnicodeDecodeError", tasks[0]["error"])

    def test_repeated_run_skips_completed_tasks(self):
        """成功后再次调用不会重复加载或重复保存。"""
        tasks = create_import_tasks([("ok.txt", b"Ready")])
        list(batch_import(tasks, self.raw_dir))
        with patch("src.data_loader.load_document") as loader:
            self.assertEqual(list(batch_import(tasks, self.raw_dir)), [{"completed": 0, "total": 0}])
            list(batch_import(tasks, self.raw_dir, retry_failed=True))
            loader.assert_not_called()
        self.assertEqual(tasks[0]["attempts"], 1)

    def test_interrupted_loading_task_can_resume(self):
        """页面中断后遗留的处理中任务可继续处理。"""
        tasks = create_import_tasks([("ok.txt", b"Ready")])
        events = batch_import(tasks, self.raw_dir)
        next(events)
        next(events)
        events.close()
        self.assertEqual(tasks[0]["status"], "loading")
        list(batch_import(tasks, self.raw_dir))
        self.assertEqual(tasks[0]["status"], "success")

    def test_empty_batch_and_duplicate_inputs(self):
        """空批次不创建目录，相同文件名和内容的重复项合并。"""
        self.assertEqual(list(batch_import([], self.raw_dir)), [{"completed": 0, "total": 0}])
        self.assertFalse(self.raw_dir.exists())
        tasks = create_import_tasks([("same.txt", b"Same"), ("same.txt", b"Same")])
        self.assertEqual(len(tasks), 1)

    def test_same_name_different_content_is_preserved(self):
        """不同内容的同名文献独立保存，不覆盖旧内容。"""
        tasks = create_import_tasks([("same.txt", b"First"), ("same.txt", b"Second")])
        list(batch_import(tasks, self.raw_dir))
        self.assertNotEqual(tasks[0]["path"], tasks[1]["path"])
        self.assertEqual([Path(t["path"]).read_bytes() for t in tasks], [b"First", b"Second"])

    def test_unsupported_format_and_unsafe_names(self):
        """路径名和不支持格式记录失败，不写到导入目录之外。"""
        names = ["../escape.txt", "/tmp/escape.txt", "folder\\escape.txt", "paper.exe"]
        tasks = create_import_tasks([(name, b"Text") for name in names])
        list(batch_import(tasks, self.raw_dir))
        self.assertTrue(all(t["status"] == "failed" for t in tasks))
        self.assertFalse(self.raw_dir.exists())

    def test_size_limit(self):
        """前端之外的调用也核验单份文档大小。"""
        tasks = create_import_tasks([("large.txt", b"A" * (1024 * 1024 + 1))])
        list(batch_import(tasks, self.raw_dir, max_file_size_mb=1))
        self.assertEqual(tasks[0]["status"], "failed")
        self.assertIn("文件过大", tasks[0]["error"])

    def test_existing_file_with_different_bytes_is_not_overwritten(self):
        """已有保存位置被外部修改时明确失败，保留其内容。"""
        tasks = create_import_tasks([("paper.txt", b"Original")])
        list(batch_import(tasks, self.raw_dir))
        path = Path(tasks[0]["path"])
        path.write_bytes(b"Changed externally")
        tasks = create_import_tasks([("paper.txt", b"Original")])
        list(batch_import(tasks, self.raw_dir))
        self.assertEqual(tasks[0]["status"], "failed")
        self.assertEqual(path.read_bytes(), b"Changed externally")

    def test_storage_failure_is_retryable(self):
        """保存位置暂时不可用时记录错误，恢复后可重试成功。"""
        self.raw_dir.write_bytes(b"not a directory")
        tasks = create_import_tasks([("ok.txt", b"Text")])
        list(batch_import(tasks, self.raw_dir))
        self.assertEqual(tasks[0]["status"], "failed")
        self.raw_dir.unlink()
        list(batch_import(tasks, self.raw_dir, retry_failed=True))
        self.assertEqual(tasks[0]["status"], "success")


class TestImportFrontend(unittest.TestCase):
    """用 Streamlit 的真实上传组件驱动页面，隔离保存目录。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        config = {"app": {"name": "智能科研助理", "description": "导入测试"},
                  "paths": {"raw_documents": self.directory.name},
                  "importing": {"max_file_size_mb": 20}}
        configuration = patch("src.utils.config.load_config", return_value=config)
        configuration.start()
        self.addCleanup(configuration.stop)
        from streamlit.testing.v1 import AppTest
        app_path = Path(__file__).resolve().parents[1] / "src/frontend/app.py"
        # 新环境首次加载界面依赖较慢，避免默认 3 秒等待导致误报。
        self.app = AppTest.from_file(str(app_path), default_timeout=10).run()

    def test_batch_upload_progress_state_and_rerun(self):
        """实际操作上传/按钮，核验混合结果、完整进度和重跑不重复执行。"""
        app = self.app
        self.assertTrue(app.button(key="start_import").disabled)
        app.file_uploader[0].set_value([
            ("good.txt", "中文正文".encode("utf-8"), "text/plain"),
            ("bad.txt", b"\xff", "text/plain"),
        ]).run()
        app.button(key="start_import").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(list(app.dataframe[0].value["状态"]), ["成功", "失败"])
        self.assertEqual(app.session_state["import_progress"], {"completed": 2, "total": 2})
        self.assertEqual(app.get("progress")[0].proto.value, 100)
        self.assertTrue(app.button(key="start_import").disabled)
        self.assertFalse(app.button(key="retry_import").disabled)
        app.button(key="retry_import").click().run()
        self.assertEqual([t["attempts"] for t in app.session_state["import_tasks"]], [1, 2])
        app.run()
        self.assertEqual([t["attempts"] for t in app.session_state["import_tasks"]], [1, 2])

    def test_failed_upload_recovers_and_disables_retry(self):
        """页面通过失败按钮恢复成功，之后重试按钮禁用且文献存在。"""
        app = self.app
        app.file_uploader[0].set_value([("retry.txt", b"Retry", "text/plain")]).run()
        with patch("src.data_loader.load_document", side_effect=OSError("暂时失败")):
            app.button(key="start_import").click().run()
        self.assertEqual(app.session_state["import_tasks"][0]["status"], "failed")
        app.button(key="retry_import").click().run()
        self.assertFalse(app.exception)
        task = app.session_state["import_tasks"][0]
        self.assertEqual(task["status"], "success")
        self.assertEqual(task["attempts"], 2)
        self.assertEqual(task["error"], "")
        self.assertTrue(Path(task["path"]).is_file())
        self.assertTrue(app.button(key="retry_import").disabled)


if __name__ == "__main__":
    unittest.main()
