"""5.1.4 混合检索与重排序：TestReranker、TestLocalReranker。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os
import tempfile
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from src.retrieval.reranker import Reranker, get_reranker


class TestReranker(unittest.TestCase):
    """大模型隔离，核验成对输入、排序与来源，不将模拟分数当实验效果。"""

    def setUp(self):
        from src.utils.config import load_config
        self.config = load_config()
        self.config["retrieval"]["top_k"] = 2
        self.config["retrieval"]["reranker_batch_size"] = 2
        patcher = patch("src.retrieval.reranker.load_config", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.reranker = Reranker()
        self.candidates = [(Document(page_content=text, metadata={
            "chunk_id": str(i), "source_file": f"论文{i}.pdf", "page_number": i + 1}), 10 - i)
            for i, text in enumerate(("无关材料", "注意力机制", "Attention mechanism"))]

    def test_pairs_model_scores_order_and_sources(self):
        """只用模型分数精排，同分保持原候选顺序，正文和来源不修改。"""
        original = [(doc.page_content, dict(doc.metadata)) for doc, _ in self.candidates]
        with patch("src.retrieval.reranker.get_reranker") as model:
            model.return_value.predict.return_value = [0.1, 0.9, 0.9]
            found = self.reranker.rerank("注意力是什么？", self.candidates)
            model.return_value.predict.assert_called_once_with(
                [["注意力是什么？", doc.page_content] for doc, _ in self.candidates],
                batch_size=2, show_progress_bar=False)
        self.assertEqual(found, [(self.candidates[1][0], 0.9), (self.candidates[2][0], 0.9)])
        self.assertIs(found[0][0], self.candidates[1][0])
        self.assertEqual(original, [(doc.page_content, doc.metadata) for doc, _ in self.candidates])

    def test_single_candidate_zero_score_and_large_k(self):
        """单候选结果仍为列表，零分保留，K 超出候选数不补齐或重复。"""
        with patch("src.retrieval.reranker.get_reranker") as model:
            model.return_value.predict.return_value = [0.0]
            self.assertEqual(self.reranker.rerank("query", self.candidates[:1], k=100),
                             [(self.candidates[0][0], 0.0)])

    def test_empty_query_or_candidates_do_not_load_model(self):
        with patch("src.retrieval.reranker.get_reranker") as model:
            self.assertEqual(self.reranker.rerank("  \n", self.candidates), [])
            self.assertEqual(self.reranker.rerank("query", []), [])
            model.assert_not_called()

    def test_invalid_k_is_rejected_before_loading_model(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(k=value), patch("src.retrieval.reranker.get_reranker") as model:
                with self.assertRaisesRegex(ValueError, "k"):
                    self.reranker.rerank("query", self.candidates, k=value)
                model.assert_not_called()

    def test_invalid_batch_and_token_length_are_rejected(self):
        for key in ("reranker_batch_size", "reranker_max_length"):
            old = self.config["retrieval"][key]
            for value in (0, -1, True, 1.5):
                with self.subTest(key=key, value=value):
                    self.config["retrieval"][key] = value
                    with self.assertRaisesRegex(ValueError, key):
                        Reranker()
            self.config["retrieval"][key] = old

    def test_score_count_and_nonfinite_values_are_rejected(self):
        for scores in ([0.1], [float("nan"), 0.1, 0.2], [float("inf"), 0.1, 0.2]):
            with self.subTest(scores=scores), patch("src.retrieval.reranker.get_reranker") as model:
                model.return_value.predict.return_value = scores
                with self.assertRaisesRegex(ValueError, "分数"):
                    self.reranker.rerank("query", self.candidates)

    def test_inference_error_is_not_silently_replaced_by_rrf(self):
        with patch("src.retrieval.reranker.get_reranker") as model:
            model.return_value.predict.side_effect = RuntimeError("推理失败")
            with self.assertRaisesRegex(RuntimeError, "推理失败"):
                self.reranker.rerank("query", self.candidates)


class TestLocalReranker(unittest.TestCase):
    """核验本地权重加载、缓存与失败恢复，测试不下载真实模型。"""

    def setUp(self):
        from src.utils.config import load_config
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = load_config()
        self.config["retrieval"]["reranker_local_path"] = self.directory.name
        patcher = patch.dict(os.environ, {"HF_HOME": str(Path(self.directory.name) / "cache")})
        patcher.start()
        self.addCleanup(patcher.stop)
        get_reranker.cache_clear()
        self.addCleanup(get_reranker.cache_clear)
        patcher = patch("src.retrieval.reranker.load_config", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_local_loading_and_reuse(self):
        """配置/模型/Tokenizer 均只读本地文件，不执行远程代码，重复调用只加载一次。"""
        from torch.nn import Sigmoid
        with patch("sentence_transformers.CrossEncoder") as model:
            self.assertIs(get_reranker(), model.return_value)
            self.assertIs(get_reranker(), model.return_value)
            model.assert_called_once()
            self.assertEqual(model.call_args.args, (self.directory.name,))
            kwargs = dict(model.call_args.kwargs)
            self.assertIsInstance(kwargs.pop("default_activation_function"), Sigmoid)
            self.assertEqual(kwargs, {"device": "cpu", "max_length": 512,
                                     "local_files_only": True, "trust_remote_code": False})

    def test_missing_model_does_not_initialize_remote_client(self):
        self.config["retrieval"]["reranker_local_path"] = str(Path(self.directory.name) / "missing")
        with patch("sentence_transformers.CrossEncoder") as model:
            with self.assertRaisesRegex(FileNotFoundError, "请先按用户手册下载权重"):
                get_reranker()
            model.assert_not_called()

    def test_failed_loading_is_not_cached_and_can_retry(self):
        """损坏权重错误保留，修复后下次调用重新加载而不是缓存失败。"""
        with patch("sentence_transformers.CrossEncoder") as model:
            model.side_effect = OSError("损坏的权重")
            with self.assertRaisesRegex(OSError, "损坏的权重"):
                get_reranker()
            model.side_effect = None
            self.assertIs(get_reranker(), model.return_value)
            self.assertEqual(model.call_count, 2)


if __name__ == "__main__":
    unittest.main()
