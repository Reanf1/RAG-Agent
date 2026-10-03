"""5.1.3 向量化与存储：TestLocalEmbeddings、TestEmbeddingEvaluation。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tempfile
import unittest
from importlib import import_module
from unittest.mock import patch
from src.retrieval.vector_store import get_embeddings


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
        evaluate_rankings = import_module("reports.5_1_3 向量化与存储.compare_embeddings").evaluate_rankings

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
        evaluate_rankings = import_module("reports.5_1_3 向量化与存储.compare_embeddings").evaluate_rankings

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


if __name__ == "__main__":
    unittest.main()
