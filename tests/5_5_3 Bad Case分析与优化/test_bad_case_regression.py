"""以真实F001和论文名误传ID故障验证修复边界；mock不是质量评测数据。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from copy import deepcopy
from io import BytesIO
import json
import unittest
from unittest.mock import patch
from langchain_core.tools import tool
from src.agent.react_loop import act, observe, run_react, _observe_events
from src.agent.tools import AVAILABLE_TOOLS
from tests.helpers import isolated_agent_logs


def packet(content=None, name=None, args=None):
    """沿用上游隔离HTTP的方式，业务参数校验与工具执行保持真实。"""
    message = {"content": json.dumps(content, ensure_ascii=False)}
    if name:
        message = {"tool_calls": [{"function": {"name": name, "arguments": args}}]}
    return BytesIO(json.dumps({"model": "qwen2.5:7b", "done": True, "done_reason": "stop",
                              "message": message, "prompt_eval_count": 100, "eval_count": 20}).encode())


class TestCitationRetention(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((ROOT / "reports/5_5_3 Bad Case分析与优化/F001引用丢失复现样例.json").read_text())
        self.context = {"observations": [deepcopy(self.fixture["tool_result"])]}

    def decision(self, context=None, decision=None):
        with patch("src.agent.react_loop.urlopen", return_value=packet(decision or self.fixture["observation"])):
            return observe(self.fixture["question"], [], context or self.context)

    def test_real_f001_preserves_source_filename_and_physical_pages(self):
        result = self.decision()
        self.assertEqual(result["answer"], self.fixture["tool_result"]["result"]["answer"])
        self.assertIn("vit.pdf；第4页（物理页码）", result["answer"])
        self.assertEqual(result["usage"], {"prompt_eval_count": 100, "eval_count": 20})

    def test_partial_or_different_question_keeps_observation_answer(self):
        for args in ({"question": "仅回答一个子问题"}, {"question": self.fixture["question"], "doc_id": "a" * 64}):
            # 显式限定论文仍是完整原问题，只有不同问题不能直通。
            context = deepcopy(self.context)
            context["observations"][0]["args"] = args
            result = self.decision(context)
            expected = self.fixture["observation"]["answer"] if args["question"] != self.fixture["question"] else self.fixture["tool_result"]["result"]["answer"]
            self.assertEqual(result["answer"], expected)

    def test_multi_tool_or_multi_step_never_substitutes_single_answer(self):
        context = deepcopy(self.context)
        context["observations"].insert(0, {"name": "paper_list", "status": "success", "result": {}})
        self.assertEqual(self.decision(context)["answer"], self.fixture["observation"]["answer"])

    def test_missing_citations_or_fallback_keeps_observation_answer(self):
        for updates in ({"citations": []}, {"generation_mode": "fallback"}, {"answer": ""}):
            with self.subTest(updates=updates):
                context = deepcopy(self.context)
                context["observations"][0]["result"].update(updates)
                self.assertEqual(self.decision(context)["answer"], self.fixture["observation"]["answer"])

    def test_incomplete_decision_is_not_turned_into_success(self):
        decision = {**self.fixture["observation"], "task_complete": False}
        result = self.decision(decision=decision)
        self.assertFalse(result["task_complete"])
        self.assertEqual(result["answer"], decision["answer"])

    def test_insufficient_evidence_cannot_be_marked_complete(self):
        context = deepcopy(self.context)
        context["observations"][0]["result"]["status"] = "insufficient_evidence"
        with self.assertRaises(RuntimeError):
            self.decision(context)


class TestPaperIdentifiers(unittest.TestCase):
    def setUp(self):
        self.invocations = []
        self.ids = ["a" * 64, "b" * 64]

        @tool
        def paper_compare(paper_a_id: str, paper_b_id: str) -> dict:
            """比对真实ID，错误名称不得当作指纹。"""
            if paper_a_id not in self.ids or paper_b_id not in self.ids:
                raise ValueError("doc_id必须是已上传论文的SHA-256指纹")
            self.invocations.append((paper_a_id, paper_b_id))
            return {"status": "answered"}

        @tool
        def paper_list() -> dict:
            """返回可核对的文献列表。"""
            self.invocations.append("list")
            return {"papers": [{"doc_id": self.ids[0], "source_file": "vit.pdf"},
                               {"doc_id": self.ids[1], "source_file": "dino.pdf"}]}

        self.tools = [paper_compare, paper_list]
        @tool
        def knowledge_base_search(question: str, doc_id: str | None = None) -> dict:
            """记录检索过滤，确保使用文献列表中的真实指纹。"""
            self.invocations.append(doc_id)
            return {"status": "answered"}
        self.tools.append(knowledge_base_search)
        self.context = {"observations": [{"name": "paper_list", "status": "success", "result": paper_list.invoke({})}]}
        self.invocations.clear()
        self.thought = {"thought": "对比论文。", "next_step": "tool", "tool_name": "paper_compare"}

    def action(self, a, b, context=None):
        with patch("src.agent.react_loop.urlopen", return_value=packet(name="paper_compare", args={"paper_a_id": a, "paper_b_id": b})):
            return list(act("对比论文", self.thought, self.tools, context or self.context))

    def test_exact_filename_and_stem_resolve_only_from_successful_list(self):
        original = deepcopy(self.context)
        events = self.action("ViT", "dino.pdf")
        self.assertEqual(self.invocations, [tuple(self.ids)])
        self.assertEqual(events[0]["args"], dict(zip(("paper_a_id", "paper_b_id"), self.ids)))
        self.assertEqual(events[-1]["status"], "success")
        self.assertEqual(original, self.context)

    def test_unknown_or_ambiguous_alias_is_not_guessed(self):
        for alias, extra in (("Vision Transformer", []), ("vit", [{"doc_id": "c" * 64, "source_file": "vit.docx"}])):
            context = deepcopy(self.context)
            context["observations"][0]["result"]["papers"].extend(extra)
            events = self.action(alias, "dino.pdf", context)
            self.assertEqual(events[-1]["status"], "error")
            self.assertEqual(self.invocations, [])

    def test_failed_list_does_not_authorize_alias_resolution(self):
        context = deepcopy(self.context)
        context["observations"][0]["status"] = "error"
        self.assertEqual(self.action("vit.pdf", "dino.pdf", context)[-1]["status"], "error")

    def test_real_identifiers_are_not_replaced(self):
        self.assertEqual(self.action(*self.ids)[-1]["status"], "success")
        self.assertEqual(self.invocations, [tuple(self.ids)])

    def test_missing_ids_plan_lists_before_comparing(self):
        decision = {"type": "observation", "observation": "尚需比较", "decision": "finish", "task_complete": False,
                    "answer": "尚未比较", "usage": {"prompt_eval_count": 0, "eval_count": 0}}
        with isolated_agent_logs(), patch("src.agent.react_loop.route_question", return_value=None), \
                patch("src.agent.react_loop.think", return_value=deepcopy(self.thought)), \
                patch("src.agent.react_loop.urlopen", return_value=packet(name="paper_list", args={})), \
                patch("src.agent.react_loop.observe", return_value=decision):
            events = list(run_react("对比ViT和DINO", self.tools))
        self.assertEqual(events[0]["tool_name"], "paper_list")
        self.assertEqual(self.invocations, ["list"])

    def test_knowledge_filename_resolves_from_real_list_and_keeps_filter(self):
        thought = {"next_step": "tool", "tool_name": "knowledge_base_search"}
        with patch("src.agent.react_loop.urlopen", return_value=packet(name="knowledge_base_search",
                   args={"question": "ViT的位置嵌入？", "doc_id": "vit.pdf"})):
            events = list(act("根据vit.pdf回答", thought, self.tools, self.context))
        self.assertEqual(self.invocations, [self.ids[0]])
        self.assertEqual(events[0]["args"]["doc_id"], self.ids[0])

    def test_knowledge_filename_with_history_lists_before_action(self):
        thought = {"next_step": "tool", "tool_name": "knowledge_base_search"}
        decision = {"type": "observation", "observation": "已取得列表", "decision": "finish",
                    "task_complete": False, "answer": "待检索"}
        with isolated_agent_logs(), patch("src.agent.react_loop.think", return_value=thought), \
                patch("src.agent.react_loop.urlopen", return_value=packet(name="paper_list", args={})), \
                patch("src.agent.react_loop.observe", return_value=decision):
            events = list(run_react("根据已上传的vit.pdf回答", self.tools,
                                   {"history": [{"role": "user", "content": "以前的问题"}]}))
        self.assertEqual(events[0]["tool_name"], "paper_list")
        self.assertEqual(self.invocations, ["list"])

    def test_knowledge_unknown_or_failed_list_cannot_authorize_filename(self):
        thought = {"next_step": "tool", "tool_name": "knowledge_base_search"}
        for alias, context in (("missing.pdf", self.context), ("vit.pdf", {"observations": []})):
            with patch("src.agent.react_loop.urlopen", return_value=packet(name="knowledge_base_search",
                       args={"question": "问题", "doc_id": alias})):
                events = list(act("根据论文回答", thought, self.tools, context))
            self.assertEqual(events[-1]["type"], "error")
            self.assertEqual(self.invocations, [])


class TestStructuredToolRetention(unittest.TestCase):
    """只对明确单任务采用工具的结构化答案；组合或证据不完整不冒充完成。"""
    def setUp(self):
        self.summary = {"status": "answered", "answer": "背景、方法、结果、结论与原文引用",
                        "sections": {key: {"text": key, "reference_ids": [1]} for key in
                                     ("background", "method", "results", "conclusion")},
                        "citations": [{"id": 1, "metadata": {"doc_id": "a" * 64}}], "missing_fields": [], "warnings": []}
        self.compare = {"status": "answered", "answer": "方法、数据集、实验结果的原文表格",
                        "comparison": [{"dimension": key, "a": {"id": 1}, "b": {"id": 2}}
                                       for key in ("方法", "数据集", "实验结果")],
                        "citations": [{"metadata": {"doc_id": identifier * 64}} for identifier in ("a", "b")],
                        "missing_dimensions": [], "low_relevance_dimensions": [], "warnings": []}

    def context(self, name, result):
        return {"observations": [{"name": name, "status": "success", "result": deepcopy(result)}]}

    def test_complete_summary_finishes_without_model_continuation(self):
        response = {"observation": "继续", "decision": "continue", "task_complete": False, "answer": ""}
        with patch("src.agent.react_loop.urlopen", return_value=packet(response)) as http:
            event = observe("生成论文的结构化摘要", AVAILABLE_TOOLS, self.context("paper_summary", self.summary))
        self.assertEqual((event["decision"], event["task_complete"], event["answer"]), ("finish", True, self.summary["answer"]))
        self.assertEqual(event["usage"], {"prompt_eval_count": 0, "eval_count": 0})
        http.assert_not_called()

    def test_comparison_keeps_all_dimensions_and_limitations(self):
        self.compare["warnings"] = ["结果段未给出精确数值，不能比较优劣。"]
        response = {"observation": "完成", "decision": "finish", "task_complete": True, "answer": "简短概述丢失维度"}
        with patch("src.agent.react_loop.urlopen", return_value=packet(response)):
            event = observe("对比两篇论文的方法、数据集、实验结果", AVAILABLE_TOOLS, self.context("paper_compare", self.compare))
        self.assertIn(self.compare["answer"], event["answer"])
        self.assertIn("未给出精确数值", event["answer"])

    def test_confirmed_compare_call_preserves_report_without_second_generation(self):
        """调用＋实验结果曾误入模型观察，真实报告超预算；确认结果应直接保留。"""
        self.compare.update(confirmed=True, low_relevance_dimensions=["论文A：方法"])
        self.compare["answer"] = "方法、数据集、定量结果及对应引用。" * 1000
        question = "请调用 paper_compare，对比两篇已入库论文的方法、数据集和实验结果"
        with patch("src.agent.react_loop.urlopen", return_value=packet({
                "observation": "改写", "decision": "finish", "task_complete": True, "answer": "报告被改写"})) as http:
            event = observe(question, AVAILABLE_TOOLS, self.context("paper_compare", self.compare))
        self.assertTrue(event["task_complete"])
        self.assertIn(self.compare["answer"], event["answer"])
        self.assertIn("低相关性维度需核验", event["answer"])
        http.assert_not_called()

    def test_missing_dimension_keeps_report_but_is_incomplete(self):
        self.compare.update(missing_dimensions=["论文B：实验结果"], status="insufficient_evidence")
        response = {"observation": "资料不足", "decision": "finish", "task_complete": False, "answer": "资料不足"}
        with patch("src.agent.react_loop.urlopen", return_value=packet(response)):
            event = observe("对比两篇论文", AVAILABLE_TOOLS, self.context("paper_compare", self.compare))
        self.assertFalse(event["task_complete"])
        self.assertIn("论文B：实验结果", event["answer"])
        self.assertIn(self.compare["answer"], event["answer"])

    def test_combined_task_and_unrelated_tool_do_not_early_finish(self):
        response = {"observation": "还需查时间", "decision": "continue", "task_complete": False, "answer": ""}
        for question, extra in (("生成结构化摘要并告诉我当前时间", []),
                                ("生成结构化摘要", [{"name": "current_time", "status": "success", "result": {}}])):
            context = self.context("paper_summary", self.summary)
            context["observations"] += extra
            with patch("src.agent.react_loop.urlopen", return_value=packet(response)) as http:
                event = observe(question, AVAILABLE_TOOLS, context)
            self.assertEqual(event["decision"], "continue")
            http.assert_called_once()

    def test_streaming_summary_after_list_has_same_final_answer(self):
        context = self.context("paper_summary", self.summary)
        context["observations"].insert(0, {"name": "paper_list", "status": "success", "result": {"papers": []}})
        with patch("src.agent.react_loop.urlopen") as http:
            events = list(_observe_events("生成结构化摘要", AVAILABLE_TOOLS, context, stream=True))
        self.assertEqual(events[-1]["answer"], self.summary["answer"])
        self.assertEqual(events[0]["type"], "token")
        http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
