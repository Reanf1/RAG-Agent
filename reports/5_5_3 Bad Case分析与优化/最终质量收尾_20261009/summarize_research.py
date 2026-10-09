"""按冻结轨迹复算题型工具选择与调用顺序，不将历史实验冒充新成绩。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCES = ROOT / "reports/5_5_2 系统性能评估/Windows正式性能评测_20261007"


def summarize(rows):
    """保留无工具请求和失败；同工具重复调用与不同工具组合分别统计。"""
    categories = {}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        correct = sum(row["tool_selection_correct"] for row in selected)
        categories[category] = {"correct": correct, "total": len(selected),
                                "accuracy": correct / len(selected)}
    sequences = Counter()
    distinct_combinations = Counter()
    for row in rows:
        names = [call["name"] for call in row["tool_calls"]]
        sequences[" → ".join(names) or "无工具"] += 1
        if len(set(names)) > 1:
            distinct_combinations[" + ".join(sorted(set(names)))] += 1
    return {"requests": len(rows), "by_category": categories,
            "sequences": dict(sequences), "distinct_tool_combinations": dict(distinct_combinations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-agent", type=Path, help="单独复算新轮完整120条，不混入历史实验")
    parser.add_argument("--output", type=Path, help="新轮必须指定全新结果路径")
    args = parser.parse_args()
    if args.current_agent:
        if not args.output or args.output.exists():
            raise ValueError("新轮需要不存在的输出路径")
        data = json.loads(args.current_agent.read_text(encoding="utf-8"))
        assert data["status"] == "completed" and len(data["rows"]) == 120
        result = {"scope": "本轮120条真实最终答案对应轨迹；工具路径不等于答案质量。",
                  "source_sha256": hashlib.sha256(args.current_agent.read_bytes()).hexdigest(),
                  "profiles": {}}
        for profile in ("default", "no_rules"):
            rows = [r for r in data["rows"] if r["profile"] == profile]
            assert len(rows) == 60
            summary = summarize(rows)
            # 执行成功只表示工具接口返回；模型标记完成也不代表语义完整。
            combinations = []
            for row in rows:
                if len({c["name"] for c in row["tool_calls"]}) > 1:
                    combinations.append({"id": row["id"], "stop_reason": row["stop_reason"],
                                         "results": [{"name": e["name"], "status": e["status"],
                                                      "business_status": e.get("result", {}).get("status")
                                                      if isinstance(e.get("result"), dict) else None}
                                                     for e in row["metrics"]["trace"] if e["type"] == "tool_result"]})
            summary["combination_outcomes"] = combinations
            result["profiles"][profile] = summary
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result["profiles"], ensure_ascii=False, indent=2))
        return
    results = {"scope": "仅复算2026-10-07的冻结轨迹；两项实验分别统计，不合并重复论文题。",
               "definition": "工具选择沿用原预先约定的可接受路径，不等于参数、任务完成或答案语义正确。",
               "sources": {}, "experiments": {}}
    for filename, profiles in [("agent.json", ("default", "no_rules")),
                               ("routing.json", ("agent",))]:
        path = SOURCES / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        results["sources"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for profile in profiles:
            rows = [row for row in data["rows"] if row["profile"] == profile]
            results["experiments"][filename + ":" + profile] = summarize(rows)
    (HERE / "研究问题分组复算.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results["experiments"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
