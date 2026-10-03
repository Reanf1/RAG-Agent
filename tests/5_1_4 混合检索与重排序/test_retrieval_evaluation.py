"""5.1.4 混合检索与重排序：TestRetrievalEvaluation。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import unittest
from importlib import import_module


class TestRetrievalEvaluation(unittest.TestCase):
    """手算 Top-5 边界、相关块集合与宏平均，实验质量来自真实模型。"""

    def test_rank_five_counts_and_rank_six_does_not(self):
        evaluate_ranking = import_module("reports.5_1_4 混合检索与重排序.compare_retrieval").evaluate_ranking
        ranking = [str(i) for i in range(8)]
        found = evaluate_ranking(ranking, ["4", "7"])
        self.assertTrue(found["hit_at_5"])
        self.assertEqual(found["recall_at_5"], 0.5)
        self.assertEqual(found["mrr_at_5"], 1 / 5)
        self.assertEqual(found["first_relevant_rank"], 5)
        self.assertFalse(evaluate_ranking(ranking, ["5"])["hit_at_5"])
        self.assertEqual(evaluate_ranking(ranking, ["5"])["mrr_at_5"], 0)

    def test_multiple_relevant_short_and_empty_results(self):
        evaluate_ranking = import_module("reports.5_1_4 混合检索与重排序.compare_retrieval").evaluate_ranking
        found = evaluate_ranking(["a", "b", "c"], ["b", "c", "z"])
        self.assertEqual(found["relevant_returned"], 2)
        self.assertEqual(found["recall_at_5"], 2 / 3)
        self.assertEqual(found["mrr_at_5"], 1 / 2)
        self.assertEqual(evaluate_ranking([], ["a"])["recall_at_5"], 0)

    def test_empty_annotations_and_duplicate_results_are_errors(self):
        evaluate_ranking = import_module("reports.5_1_4 混合检索与重排序.compare_retrieval").evaluate_ranking
        for ranking, labels in ((["a"], []), (["a", "a"], ["a"])):
            with self.subTest(ranking=ranking), self.assertRaises(ValueError):
                evaluate_ranking(ranking, labels)

    def test_unequal_groups_and_repeated_timing_are_not_duplicate_queries(self):
        evaluate_ranking = import_module("reports.5_1_4 混合检索与重排序.compare_retrieval").evaluate_ranking
        summarize = import_module("reports.5_1_4 混合检索与重排序.compare_retrieval").summarize
        rows = [{**evaluate_ranking(ids, ["a"]), "group": group, "latency_ms_runs": times}
                for ids, group, times in ((["a"], "zh->zh", [10, 20]),
                                         (["b"], "zh->zh", [30, 40]),
                                         (["a"], "en->en", [50, 60]))]
        result = summarize(rows)
        self.assertEqual(result["query_count"], 3)
        self.assertEqual(result["hit_count"], 2)
        self.assertEqual(result["hit_at_5"], 2 / 3)
        self.assertEqual(result["macro"]["hit_at_5"], 0.75)
        self.assertEqual(result["worst_group_hit_at_5"], 0.5)
        self.assertEqual(result["latency_mean_ms"], 35)
        self.assertEqual(result["latency_p95_ms"], 57.5)

    def test_sample_ids_and_annotation_membership_are_validated(self):
        validate_sample = import_module("reports.5_1_4 混合检索与重排序.compare_retrieval").validate_sample
        sample = {"corpus": [{"id": "a"}], "queries": [{"id": "q", "relevant_ids": ["a"]}]}
        validate_sample(sample)
        sample["queries"][0]["relevant_ids"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "相关标注"):
            validate_sample(sample)
        sample["queries"][0]["relevant_ids"] = ["a"]
        sample["corpus"].append({"id": "a"})
        with self.assertRaisesRegex(ValueError, "重复 ID"):
            validate_sample(sample)


if __name__ == "__main__":
    unittest.main()
