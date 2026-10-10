"""从逐题排名和实际HTTP响应独立复算指标，失败回答仍保留在样本中。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/raw/evaluation_vision_transformers/corpus.json",
                        help="与本轮结果对应的本机冻结语料，按原始字节严格核验")
    parser.add_argument("--retrieval-preparation-only", action="store_true", help="检索阶段仅1题建库检查，Agent仍须60题两组")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("复核报告不能覆盖历史结果")
    retrieval, agent = [json.loads(p.read_text(encoding="utf-8")) for p in (args.retrieval, args.agent)]
    assert retrieval["status"] == agent["status"] == "completed"
    assert agent["question_count"] == 60
    assert retrieval["question_count"] == (1 if args.retrieval_preparation_only else 60)
    assert retrieval["inputs"] == agent["inputs"]
    manifest = json.loads((ROOT / "reports/5_5_1 评测集构建/论文清单.json").read_text(encoding="utf-8"))
    ids = {p["id"]: p["doc_id"] for p in manifest["papers"]}
    questions = {q["id"]: q for q in json.loads((ROOT / "reports/评测集.json").read_text(encoding="utf-8"))}
    for key, path in (("dataset_sha256", ROOT / "reports/评测集.json"),
                      ("manifest_sha256", ROOT / "reports/5_5_1 评测集构建/论文清单.json"),
                      ("corpus_sha256", args.corpus),
                      ("config_sha256", ROOT / "config.yaml"),
                      ("evaluation_script_sha256", Path(__file__).with_name("evaluate_system.py"))):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == retrieval["inputs"][key], key
    for name, expected in retrieval["inputs"]["source_sha256"].items():
        # Windows归档用反斜杠，在Mac复核同一相对源码路径时统一分隔符。
        assert hashlib.sha256((ROOT / name.replace("\\", "/")).read_bytes()).hexdigest() == expected, name
    assert len(retrieval["rows"]) == (5 if args.retrieval_preparation_only else 300) and len(agent["rows"]) == 120
    retrieval_ids = {next(iter(questions))} if args.retrieval_preparation_only else questions.keys()
    for key in retrieval["profiles"]:
        rows = [r for r in retrieval["rows"] if r["profile"] == key]
        assert {r["id"] for r in rows} == retrieval_ids and len(rows) == retrieval["question_count"]
        for row in rows:
            gold = {(ids[e["paper_id"]], e["page_number"]) for e in questions[row["id"]]["evidence"]}
            hits, ranks = set(), []
            assert len(row["top5"]) <= 5
            assert len({x["metadata"]["chunk_id"] for x in row["top5"]}) == len(row["top5"])
            for index, chunk in enumerate(row["top5"], 1):
                m = chunk["metadata"]
                matched = {(pid, page) for pid, page in gold if pid == m["doc_id"] and
                           m["page_number"] <= page <= m.get("page_end", m["page_number"])}
                if matched:
                    ranks.append(index)
                hits.update(matched)
            assert row["hit_at_5"] == bool(hits)
            assert row["mrr_at_5"] == (1 / min(ranks) if ranks else 0)
            assert row["recall_at_5"] == len(hits) / len(gold)
            assert row["paper_coverage_at_5"] == len({pid for pid, page in hits}) / len({pid for pid, page in gold})
            assert row["all_papers_hit_at_5"] == ({pid for pid, page in hits} == {pid for pid, page in gold})
        for metric in ("hit_at_5", "mrr_at_5", "recall_at_5", "paper_coverage_at_5", "all_papers_hit_at_5"):
            assert abs(retrieval["profiles"][key][metric] - statistics.mean(r[metric] for r in rows)) < 1e-12
        assert abs(retrieval["profiles"][key]["latency_mean_ms"] - statistics.mean(r["seconds"] * 1000 for r in rows)) < 1e-9
    calls = [json.loads(line) for line in args.agent.with_suffix(".calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(c["request"]["options"]["seed"] == 20261003 for c in calls)
    for key in agent["profiles"]:
        rows = [r for r in agent["rows"] if r["profile"] == key]
        assert {r["id"] for r in rows} == questions.keys() and len(rows) == 60
        for row in rows:
            # 从实际轨迹复核选择路径，避免只信任逐题布尔值的均值。
            tools = [e for e in row["metrics"]["trace"] if e["type"] == "tool_call"]
            assert [{k: e[k] for k in ("name", "args", "iteration")} for e in tools] == row["tool_calls"]
            q = questions[row["id"]]
            names = {e["name"] for e in tools}
            allowed = {"knowledge_base_search", "paper_list"}
            if q["category"] == "comparison":
                allowed.add("paper_compare")
            compared = {e["args"].get(k) for e in tools if e["name"] == "paper_compare"
                        for k in ("paper_a_id", "paper_b_id")}
            correct = bool(names) and names <= allowed and ("knowledge_base_search" in names or
                ("paper_compare" in names and compared == {ids[p] for p in q["paper_ids"]}))
            assert row["tool_selection_correct"] == correct
            selected = [c for c in calls if c["id"] == row["id"] and c["profile"] == key]
            assert len(selected) == row["model_calls"]
            unknown = sum(any(type(c.get("response", {}).get(field)) is not int for field in
                              ("prompt_eval_count", "eval_count")) for c in selected)
            inputs = sum(c.get("response", {}).get("prompt_eval_count", 0) or 0 for c in selected)
            outputs = sum(c.get("response", {}).get("eval_count", 0) or 0 for c in selected)
            assert row["tokens"] == {"input_known": inputs, "output_known": outputs,
                "total": None if unknown else inputs + outputs, "unknown_calls": unknown, "source": "actual_ollama_responses"}
            assert row["event_types"][-1] == "done"
            assert 1 <= row["iterations"] <= 8 and row["seconds"] >= 0
            assert row["stop_reason"] in ("task_complete", "incomplete", "error", "max_iterations", "repeated_calls", "tool_timeout")
        expected = agent["profiles"][key]
        assert expected["tool_selection_accuracy"] == statistics.mean(r["tool_selection_correct"] for r in rows)
        assert expected["iterations_mean"] == statistics.mean(r["iterations"] for r in rows)
        assert expected["latency_mean_seconds"] == statistics.mean(r["seconds"] for r in rows)
        assert expected["tokens_total"] == sum(r["tokens"]["total"] or 0 for r in rows)
        assert expected["stop_reasons"] == dict(Counter(r["stop_reason"] for r in rows))
    output = {"passed": True, "retrieval_rows": len(retrieval["rows"]), "agent_rows": len(agent["rows"]),
              "retrieval_scope": "1题五配置仅用于建库核验" if args.retrieval_preparation_only else "60题五配置正式检索实验",
              "actual_model_calls": len(calls), "source_and_input_hashes_unchanged": True,
              "checks": "逐题页级匹配、排名倒数、跨论文覆盖、轨迹工具路径、均值、实际HTTP Token、失败分母、固定种子及源码哈希",
              "human_quality": "本文件只复算结构与性能指标，逐题质量分数见单独的助手量表评阅记录"}
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
