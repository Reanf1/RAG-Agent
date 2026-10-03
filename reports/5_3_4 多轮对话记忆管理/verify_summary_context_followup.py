"""复用首次失败的真实摘要Context，验证模型读取已给出的会话代号；不重写失败结果。"""

import argparse
from datetime import datetime
import json
from importlib import import_module
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

serial = import_module("reports.5_3_3 Agent决策优化.verify_error_recovery").serial
from src.agent.react_loop import run_react


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("请使用新路径保留原始证据")
    source = json.loads(args.source.read_text())
    row = next(row for row in source["rows"] if row["name"] == "long_memory_summary_followup")
    original = row["events"][-1]["context"]
    context = {key: original[key] for key in ("summary", "history", "history_window", "memory_summary")}
    code = re.search(r"EXP_[A-F0-9]+", context["summary"]).group()
    report = {"started_at": datetime.now().astimezone().isoformat(), "source": str(args.source),
              "context": context, "expected_code": code, "events": [], "passed": False}
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for event in run_react("我的当前实验代号是什么？只回答代号。", tools=[], context=context):
        report["events"].append(serial(event))
        save()
        print(event["type"], event.get("full_response", ""), flush=True)
    done = report["events"][-1]
    report["passed"] = done["task_complete"] and done["full_response"].strip() == code
    report["finished_at"] = datetime.now().astimezone().isoformat()
    save()
    if not report["passed"]:
        raise RuntimeError("读取摘要中的已有信息仍失败，真实响应已保留")


if __name__ == "__main__":
    main()
