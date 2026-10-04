"""独立复算完整60题、真实Token与源码边界，保留不可变的优化前证据。"""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "reports/5_5_2 系统性能评估"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    lock = read(HERE / "迭代方案_锁定.json")
    before = read(BASE / "Agent两组结果_20261003.json")
    after_path = HERE / "Agent优化后60题_20261003.json"
    after = read(after_path)
    assert after["status"] == "completed" and len(after["rows"]) == 60
    for name, expected in lock["baseline_sha256"].items():
        assert digest(BASE / name) == expected, name
    for name, expected in lock["inputs_sha256"].items():
        assert digest(ROOT / name) == expected, name
    assert before["inputs"]["source_sha256"] == lock["source_before"]
    assert digest(HERE / "react_loop_优化前.py.txt") == lock["source_before"]["src/agent/react_loop.py"]
    assert after["inputs"]["evaluation_script_sha256"] == digest(HERE / "evaluate_iteration.py")
    changed = []
    for name, old_hash in lock["source_before"].items():
        new_hash = digest(ROOT / name)
        assert new_hash == after["inputs"]["source_sha256"][name], name
        if old_hash != new_hash:
            changed.append(name)
    assert changed == ["src/agent/react_loop.py"], changed
    assert before["business_config"] == after["business_config"]
    assert {m["digest"] for m in before["model_tags"]["models"]} == {m["digest"] for m in after["model_tags"]["models"]}
    assert before["model_seed"] == after["model_seed"] == 20261003
    questions = {q["id"]: q for q in read(ROOT / "reports/评测集.json")}
    ids = {p["id"]: p["doc_id"] for p in read(ROOT / "reports/5_5_1 评测集构建/论文清单.json")["papers"]}
    paired = read(HERE / "优化前后对比数据.json")
    for label, run, raw_path in (("before", before, BASE / "Agent两组结果_20261003.calls.jsonl"),
                                  ("after", after, after_path.with_suffix(".calls.jsonl"))):
        rows = [r for r in run["rows"] if r["profile"] == "default"]
        calls = [json.loads(line) for line in raw_path.read_text().splitlines()]
        calls = [c for c in calls if c["profile"] == "default"]
        assert len(rows) == 60 and {r["id"] for r in rows} == set(questions)
        assert all(c["request"]["options"]["seed"] == 20261003 for c in calls)
        for row in rows:
            selected = [c for c in calls if c["id"] == row["id"]]
            assert len(selected) == row["model_calls"]
            unknown = sum(any(type(c.get("response", {}).get(k)) is not int for k in ("prompt_eval_count", "eval_count")) for c in selected)
            inputs = sum(c.get("response", {}).get("prompt_eval_count", 0) or 0 for c in selected)
            outputs = sum(c.get("response", {}).get("eval_count", 0) or 0 for c in selected)
            assert row["tokens"]["total"] == (None if unknown else inputs + outputs)
            assert row["tokens"]["input_known"] == inputs and row["tokens"]["output_known"] == outputs
            assert row["tokens"]["unknown_calls"] == unknown
            trace = row["metrics"]["trace"]
            tools = [e for e in trace if e["type"] == "tool_call"]
            assert [{k: e[k] for k in ("name", "args", "iteration")} for e in tools] == row["tool_calls"]
            names = {e["name"] for e in tools}
            allowed = {"paper_list", "knowledge_base_search"}
            if questions[row["id"]]["category"] == "comparison":
                allowed.add("paper_compare")
            compared = {e["args"].get(k) for e in tools if e["name"] == "paper_compare" for k in ("paper_a_id", "paper_b_id")}
            correct = bool(names) and names <= allowed and ("knowledge_base_search" in names or
                        ("paper_compare" in names and compared == {ids[p] for p in questions[row["id"]]["paper_ids"]}))
            assert row["tool_selection_correct"] == correct
            assert 1 <= row["iterations"] <= 8 and row["event_types"][-1] == "done"
        known = [r["tokens"]["total"] for r in rows if r["tokens"]["total"] is not None]
        metrics = paired[label]["metrics"]
        recomputed = {"tool_selection_accuracy": statistics.mean(r["tool_selection_correct"] for r in rows),
                      "iterations_mean": statistics.mean(r["iterations"] for r in rows),
                      "latency_mean_seconds": statistics.mean(r["seconds"] for r in rows),
                      "task_complete_rate": statistics.mean(r["task_complete"] for r in rows),
                      "tokens_total": sum(known), "tokens_mean_known_requests": statistics.mean(known)}
        for key, value in recomputed.items():
            assert math.isclose(metrics[key], value, rel_tol=1e-12), (label, key)
        assert all(math.isclose(run["profiles"]["default"][key], value, rel_tol=1e-12) for key, value in recomputed.items())
    assert len(paired["paired"]) == 60
    # 复核结构改进：原样保留条件独立核对；不能据此宣布语义引用质量通过。
    for label, run in (("before", before), ("after", after)):
        rows = [r for r in run["rows"] if r["profile"] == "default"]
        structural = paired[label]["structural"]
        invalid, eligible, retained = [], [], []
        for row in rows:
            for call in row["tool_calls"]:
                fields = ("paper_a_id", "paper_b_id") if call["name"] == "paper_compare" else ("doc_id",) if call["name"] in {"paper_metadata", "paper_summary"} else ()
                if fields and any(call["args"].get(k) not in ids.values() for k in fields):
                    invalid.append(row["id"])
            results = [e for e in row["metrics"]["trace"] if e["type"] == "tool_result"]
            if row["task_complete"] and len(results) == 1:
                tool = results[0]
                evidence = tool.get("result") or {}
                if (tool["name"] == "knowledge_base_search" and tool["status"] == "success"
                        and tool["args"].get("question") == questions[row["id"]]["question"]
                        and evidence.get("status") == "answered" and evidence.get("generation_mode") == "grounded"
                        and evidence.get("citations") and evidence.get("answer")):
                    eligible.append(row["id"])
                    if evidence["answer"] == row["answer"]:
                        retained.append(row["id"])
        assert len(invalid) == structural["invalid_paper_id_call_count"]
        assert eligible == structural["eligible_single_rag_questions"]
        assert retained == structural["exact_source_answer_retained"]
    evidence = {"verified_at": datetime.now().astimezone().isoformat(), "status": "passed",
                "paired_questions": 60, "source_changed": changed,
                "checks": ["旧基线字节未改、输入/配置/模型一致", "源码边界与旧源码快照", "完整题集及失败保留",
                           "实际HTTP用量与工具路径复算", "所有汇总均值", "引用直通与无效ID结构指标"],
                "human_scoring": "pending_independent_review", "files_sha256": {p.name: digest(p) for p in
                    (after_path, after_path.with_suffix(".calls.jsonl"), HERE / "优化前后对比数据.json", HERE / "Bad_Case清单.json")}}
    (HERE / "迭代数值复核.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
