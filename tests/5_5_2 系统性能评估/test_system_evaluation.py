"""用手算例子验证评分边界，避免错误口径污染真实实验结果。"""

import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("system_evaluation", ROOT / "reports/5_5_2 系统性能评估/evaluate_system.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class EvaluationMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = {q["id"]: q for q in json.loads(evaluation.DATASET.read_text(encoding="utf-8"))}
        cls.manifest = json.loads(evaluation.MANIFEST.read_text(encoding="utf-8"))
        cls.identifiers = {p["id"]: p["doc_id"] for p in cls.manifest["papers"]}

    def chunk(self, paper, page, end=None):
        metadata = {"doc_id": self.identifiers[paper], "page_number": page}
        if end is not None:
            metadata["page_end"] = end
        return {"metadata": metadata}

    def test_second_rank_has_reciprocal_half(self):
        result = evaluation.page_metrics([self.chunk("vit", 3), self.chunk("vit", 4)], self.questions["F001"])
        self.assertEqual((result["hit_at_5"], result["mrr_at_5"], result["recall_at_5"]), (True, .5, 1))

    def test_sixth_rank_does_not_count(self):
        result = evaluation.page_metrics([self.chunk("vit", 3)] * 5 + [self.chunk("vit", 4)], self.questions["F001"])
        self.assertEqual((result["hit_at_5"], result["mrr_at_5"], result["recall_at_5"]), (False, 0, 0))

    def test_same_page_is_not_counted_twice(self):
        result = evaluation.page_metrics([self.chunk("vit", 4)] * 5, self.questions["S001"])
        self.assertEqual(result["recall_at_5"], .25)
        self.assertEqual(result["paper_coverage_at_5"], .25)
        self.assertFalse(result["all_papers_hit_at_5"])

    def test_page_number_requires_same_paper(self):
        result = evaluation.page_metrics([self.chunk("deit", 4)], self.questions["F001"])
        self.assertFalse(result["hit_at_5"])

    def test_cross_page_block_includes_evidence_page(self):
        result = evaluation.page_metrics([self.chunk("vit", 3, 4)], self.questions["F001"])
        self.assertEqual(result["recall_at_5"], 1)

    def test_all_required_papers_are_covered(self):
        ranked = [self.chunk("vit", 4), self.chunk("deit", 1), self.chunk("swin", 1), self.chunk("detr", 8)]
        result = evaluation.page_metrics(ranked, self.questions["S001"])
        self.assertEqual(result["recall_at_5"], 1)
        self.assertTrue(result["all_papers_hit_at_5"])

    def test_no_tool_and_unrelated_tool_are_wrong(self):
        for calls in ([], [{"name": "calculator", "args": {}}]):
            self.assertFalse(evaluation.tool_selection(self.questions["F001"], calls, self.manifest))

    def test_paper_list_alone_does_not_answer_fact(self):
        self.assertFalse(evaluation.tool_selection(self.questions["F001"], [{"name": "paper_list", "args": {}}], self.manifest))

    def test_comparison_needs_correct_paper_ids(self):
        correct = [{"name": "paper_compare", "args": {"paper_a_id": self.identifiers["vit"], "paper_b_id": self.identifiers["deit"]}}]
        self.assertTrue(evaluation.tool_selection(self.questions["C001"], correct, self.manifest))
        wrong = [{"name": "paper_compare", "args": {"paper_a_id": self.identifiers["vit"], "paper_b_id": self.identifiers["swin"]}}]
        self.assertFalse(evaluation.tool_selection(self.questions["C001"], wrong, self.manifest))

    def test_failure_keeps_denominator_and_unknown_tokens(self):
        rows = [{"tool_selection_correct": True, "iterations": 1, "seconds": 10, "task_complete": True,
                 "tokens": {"total": 30, "input_known": 20, "output_known": 10}, "stop_reason": "task_complete"},
                {"tool_selection_correct": False, "iterations": 2, "seconds": 20, "task_complete": False,
                 "tokens": {"total": None, "input_known": 5, "output_known": 0}, "stop_reason": "error"}]
        result = evaluation.summarize_agent(rows)
        self.assertEqual(result["tool_selection_accuracy"], .5)
        self.assertEqual(result["iterations_mean"], 1.5)
        self.assertEqual(result["latency_mean_seconds"], 15)
        self.assertEqual(result["token_unknown_requests"], 1)
        self.assertEqual(result["tokens_total"], 30)


if __name__ == "__main__":
    unittest.main()
