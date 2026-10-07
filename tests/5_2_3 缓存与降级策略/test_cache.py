"""5.2.3 缓存与降级策略：TestSemanticCache。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from copy import deepcopy
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from src.generation.rag_pipeline import resolve_citations
from src.generation.cache import SemanticCache, cache_scope


class TestSemanticCache(unittest.TestCase):
    """明确向量与真实引用快照验证缓存；真实 M3E 时延和误匹配另存 reports。"""

    def setUp(self):
        from src.utils.config import load_config
        self.config = load_config()
        patcher = patch("src.generation.cache.load_config", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("src.generation.cache.get_embeddings")
        self.embedding = patcher.start().return_value
        self.addCleanup(patcher.stop)
        self.embedding.embed_query.return_value = [3.0, 4.0]
        self.cache = SemanticCache()
        context = {"references": [{"id": 1, "source_file": "attention.pdf", "location": "第3页（物理页码）",
                                  "text": "编码器有6层。", "metadata": {"chunk_id": "功能样例块"}, "truncated": False}]}
        self.result = {"type": "done", **resolve_citations("编码器有6层。[参考文档1]", context),
                       "done_reason": "stop", "model": "qwen2.5:7b", "raw_answer": "编码器有6层。[参考文档1]",
                       "usage": {"prompt_eval_count": 400, "eval_count": 20, "total_duration": 1000000000}}

    def test_empty_and_blank_requests_never_load_model(self):
        self.assertIsNone(self.cache.lookup("问题", "scope"))
        with self.assertRaisesRegex(ValueError, "不能为空"):
            self.cache.lookup(" ", "scope")
        self.embedding.embed_query.assert_not_called()

    def test_degraded_answer_is_not_cached_even_after_user_confirmation(self):
        for mode in ("empty", "low"):
            self.assertFalse(self.cache.put("问题", {**self.result, "generation_mode": mode}, "scope"))
        self.embedding.embed_query.assert_not_called()
        self.assertEqual(self.cache.entries, [])

    def test_exact_hit_skips_embedding_preserves_answer_and_uses_zero_current_tokens(self):
        self.assertTrue(self.cache.put("层数？", self.result, "scope"))
        self.embedding.embed_query.reset_mock()
        hit = self.cache.lookup(" 层数？ ", "scope")
        self.assertEqual(hit["cache"]["mode"], "exact")
        self.assertEqual(hit["answer"], self.result["answer"])
        self.assertEqual(hit["citations"], self.result["citations"])
        self.assertEqual(hit["usage"]["prompt_eval_count"], 0)
        self.assertEqual(hit["original_usage"]["prompt_eval_count"], 400)
        self.embedding.embed_query.assert_not_called()

    def test_semantic_matching_normalizes_vectors_and_selects_highest_score(self):
        self.embedding.embed_query.return_value = [1.0, 0.0]
        self.cache.put("介绍编码器层数", self.result, "scope")
        self.embedding.embed_query.return_value = [0.98, 0.2]
        self.cache.put("介绍编码器结构", self.result, "scope")
        self.embedding.embed_query.return_value = [5.0, 0.0]
        hit = self.cache.lookup("编码器包含多少层？", "scope")
        self.assertEqual(hit["cache"]["mode"], "semantic")
        self.assertEqual(hit["cache"]["question"], "介绍编码器层数")
        self.assertAlmostEqual(hit["cache"]["similarity"], 1.0)

    def test_threshold_includes_boundary_and_rejects_lower_score(self):
        self.embedding.embed_query.return_value = [1.0, 0.0]
        self.cache.put("编码器层数？", self.result, "scope")
        for score, expected in ((0.97, True), (0.969, False), (0.5, False)):
            self.embedding.embed_query.return_value = [score, (1 - score ** 2) ** 0.5]
            self.assertEqual(self.cache.lookup("编码器有几层？", "scope") is not None, expected)

    def test_changed_numbers_models_negation_and_language_never_reuse_identical_vectors(self):
        pairs = [("BERT使用15%的掩码比例吗？", "BERT使用20%的掩码比例吗？"),
                 ("BERT-base有几层？", "BERT-large有几层？"),
                 ("论文A使用了什么方法？", "论文B使用了什么方法？"),
                 ("使用多头注意力吗？", "没有使用多头注意力吗？"),
                 ("It uses attention?", "It does not use attention?"),
                 ("Answer in English: explain BERT.", "Answer in Chinese: explain BERT."),
                 ("解释注意力", "Explain attention")]
        for left, right in pairs:
            self.cache.clear()
            self.cache.put(left, self.result, "scope")
            self.embedding.embed_query.reset_mock()
            self.assertIsNone(self.cache.lookup(right, "scope"), (left, right))
            self.embedding.embed_query.assert_not_called()

    def test_long_questions_are_exact_only_to_avoid_embedding_tail_truncation(self):
        left = "原始文献问题" * 60 + "甲"
        self.cache.put(left, self.result, "scope")
        self.assertEqual(self.cache.lookup(left, "scope")["cache"]["mode"], "exact")
        self.assertIsNone(self.cache.lookup(left[:-1] + "乙", "scope"))
        self.embedding.embed_query.assert_not_called()

    def test_changed_comparison_or_numeric_sign_and_scale_never_reuses_answer(self):
        """真实M3E会忽略比较方向及百分／千分符号，即使同向量也不能复用答案。"""
        pairs = [("模型准确率是否超过80%？", "模型准确率是否低于80%？"),
                 ("模型准确率大于80%吗？", "模型准确率小于80%吗？"),
                 ("BERT使用15%的掩码比例吗？", "BERT使用15‰的掩码比例吗？"),
                 ("指标是否>=80？", "指标是否<=80？"),
                 ("指标至少80吗？", "指标至多80吗？"),
                 ("Is accuracy greater than 80%?", "Is accuracy less than 80%?"),
                 ("指标为-5吗？", "指标为5吗？")]
        for left, right in pairs:
            with self.subTest(left=left, right=right):
                self.cache.clear()
                self.cache.put(left, self.result, "scope")
                self.embedding.embed_query.reset_mock()
                self.assertIsNone(self.cache.lookup(right, "scope"))
                self.embedding.embed_query.assert_not_called()

    def test_errors_length_missing_or_invalid_sources_and_empty_answers_are_not_stored(self):
        invalid = [{"type": "error"}, {"done_reason": "length"}, {"warnings": ["未完整"]},
                   {"citations": []}, {"missing_citations": True}, {"invalid_citation_ids": [9]}, {"answer": " "}]
        for change in invalid:
            self.assertFalse(self.cache.put("问题", {**self.result, **change}, "scope"))
        self.assertFalse(self.cache.entries)
        self.embedding.embed_query.assert_not_called()

    def test_cache_input_and_returned_source_snapshots_are_independent(self):
        original = deepcopy(self.result)
        self.cache.put("问题", self.result, "scope")
        self.result["citations"][0]["text"] = "外部修改"
        hit = self.cache.lookup("问题", "scope")
        self.assertEqual(hit["citations"], original["citations"])
        hit["citations"][0]["metadata"]["chunk_id"] = "篡改"
        hit["usage"]["eval_count"] = 999
        self.assertEqual(self.cache.lookup("问题", "scope")["citations"], original["citations"])
        self.assertEqual(self.cache.lookup("问题", "scope")["original_usage"]["eval_count"], 20)

    def test_capacity_eviction_replacement_and_clear(self):
        self.config["generation"]["cache"]["max_entries"] = 2
        self.cache = SemanticCache()
        for question in ("问题1", "问题2", "问题3"):
            self.cache.put(question, self.result, "scope")
        self.assertIsNone(self.cache.lookup("问题1", "scope"))
        self.cache.put("问题2", {**self.result, "answer": "更新答案"}, "scope")
        self.assertEqual(len(self.cache.entries), 2)
        self.assertEqual(self.cache.lookup("问题2", "scope")["answer"], "更新答案")
        self.cache.clear()
        self.assertIsNone(self.cache.lookup("问题2", "scope"))

    def test_scope_change_and_session_instances_are_isolated(self):
        self.cache.put("问题", self.result, "scope-a")
        other = SemanticCache()
        self.assertIsNone(other.lookup("问题", "scope-a"))
        self.assertIsNone(self.cache.lookup("问题", "scope-b"))
        self.assertIsNone(self.cache.lookup("问题", "scope-a"))

    def test_scope_detects_add_delete_same_count_replacement_and_position_changes(self):
        from unittest.mock import Mock
        doc = Document(page_content="原文", metadata={"chunk_id": "同ID", "page_number": 3})
        store = Mock()
        store.list_chunks.return_value = [doc]
        scope = cache_scope(store)
        for docs in ([], [doc, Document(page_content="新增", metadata={"chunk_id": "新ID"})],
                     [Document(page_content="替换原文", metadata=doc.metadata)],
                     [Document(page_content=doc.page_content, metadata={**doc.metadata, "page_number": 4})]):
            store.list_chunks.return_value = docs
            self.assertNotEqual(cache_scope(store), scope)
        store.list_chunks.return_value = [doc]
        self.assertEqual(cache_scope(store), scope)
        self.embedding.embed_query.assert_not_called()

    def test_scope_ignores_corpus_order_but_invalidates_prompt_and_model_parameters(self):
        from unittest.mock import Mock
        docs = [Document(page_content=content) for content in ("甲", "乙")]
        store = Mock()
        store.list_chunks.return_value = docs
        scope = cache_scope(store)
        store.list_chunks.return_value = list(reversed(docs))
        self.assertEqual(cache_scope(store), scope)
        self.config["llm"]["temperature"] += 0.1
        self.assertNotEqual(cache_scope(store), scope)
        self.config["llm"]["temperature"] -= 0.1
        with patch("src.generation.cache.PROMPT_VERSION", "新版本"):
            self.assertNotEqual(cache_scope(store), scope)

    def test_invalid_configuration_and_vectors_are_explicit(self):
        for key, value in (("max_entries", 0), ("max_entries", True),
                           ("similarity_threshold", 0), ("similarity_threshold", 1.1),
                           ("similarity_threshold", float("nan"))):
            original = self.config["generation"]["cache"][key]
            self.config["generation"]["cache"][key] = value
            with self.assertRaises(ValueError):
                SemanticCache()
            self.config["generation"]["cache"][key] = original
        for vector in ([], [0, 0], [float("nan"), 1]):
            self.embedding.embed_query.return_value = vector
            with self.assertRaises(ValueError):
                self.cache.put("问题", self.result, "scope")


if __name__ == "__main__":
    unittest.main()
