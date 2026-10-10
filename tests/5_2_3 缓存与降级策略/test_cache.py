"""5.2.3 缓存与降级策略：TestSemanticCache。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import unittest
from unittest.mock import patch
from src.generation.rag_pipeline import resolve_citations
from src.generation.cache import SemanticCache


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


    def test_scope_change_and_session_instances_are_isolated(self):
        self.cache.put("问题", self.result, "scope-a")
        other = SemanticCache()
        self.assertIsNone(other.lookup("问题", "scope-a"))
        self.assertIsNone(self.cache.lookup("问题", "scope-b"))
        self.assertIsNone(self.cache.lookup("问题", "scope-a"))


if __name__ == "__main__":
    unittest.main()
