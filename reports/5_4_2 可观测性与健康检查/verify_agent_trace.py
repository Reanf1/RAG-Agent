"""真实本地模型与SQLite核验公开决策轨迹、工具调用成功率和耗时。"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import json
from importlib import import_module
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

serial = import_module("reports.5_3_4 多轮对话记忆管理.verify_session_isolation").serial
from src.agent import MemoryManager, run_session
from src.agent.tools import calculator, current_time
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError("请使用新输出文件，保留原核验记录")
    report = {"started_at": datetime.now().astimezone().isoformat(), "rows": [],
              "scope": "真实Qwen/Ollama、计算器、当前时间和SQLite；功能核验，不代表论文答案质量。"}

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    config = deepcopy(load_config())
    with tempfile.TemporaryDirectory(prefix="agent-trace-") as directory, ExitStack() as stack:
        config["paths"]["logs"] = str(Path(directory) / "logs")
        memory = MemoryManager(Path(directory) / "memory.sqlite3")
        for module in ("src.agent.react_loop", "src.agent.router", "src.agent.memory", "src.agent.tools", "src.utils.logger"):
            stack.enter_context(patch(module + ".load_config", return_value=config))
        for name, question in (
            ("calculator_success", "调用calculator计算3.14*2.56。"),
            ("calculator_error", "调用calculator计算1/0，记录工具返回的错误。"),
            ("independent_tools", "同时调用calculator计算2*3、调用current_time返回当前时间，这两个工具相互独立。"),
        ):
            row = {"name": name, "question": question, "events": []}
            report["rows"].append(row)
            session = memory.create_session("trace-verifier")
            for event in run_session(question, "trace-verifier", session, [calculator, current_time], memory=memory):
                row["events"].append(serial(event))
                print(name, event["type"], event["metrics"]["tools"], flush=True)
                save()
            metrics = row["events"][-1]["metrics"]
            returned = [e for e in row["events"] if e["type"] == "tool_result"]
            expected_rate = sum(e["status"] == "success" for e in returned) / len(returned) if returned else None
            expected_mean = sum(e["elapsed_seconds"] for e in returned) / len(returned) if returned else None
            row["trace_check"] = [t["type"] for t in metrics["trace"]] == [e["type"] for e in row["events"]]
            row["stats_check"] = metrics["tools"]["success_rate"] == expected_rate \
                and metrics["tools"]["mean_seconds"] == expected_mean \
                and metrics["tools"]["completed"] == len(returned) and bool(returned)
            row["scenario_check"] = any(e["status"] == ("error" if name == "calculator_error" else "success") for e in returned)
            if name == "independent_tools":
                row["scenario_check"] = {e["name"] for e in returned} == {"calculator", "current_time"}
                row["parallel_observed"] = all(e["execution_mode"] == "parallel" for e in returned)
            row["summary"] = {"tools": metrics["tools"], "response_seconds": metrics["response_seconds"],
                              "stop_reason": row["events"][-1]["stop_reason"],
                              "task_complete": row["events"][-1]["task_complete"]}
            save()
        report["records"] = [json.loads(line) for path in Path(config["paths"]["logs"]).glob("agent_*.jsonl")
                             for line in path.read_text().splitlines()]
        report["persisted_check"] = all(next(r for r in reversed(report["records"]) if r["request_id"] == row["events"][-1]["request_id"])["metrics"]
            == row["events"][-1]["metrics"] for row in report["rows"])
    report["passed"] = report["persisted_check"] and all(row["trace_check"] and row["stats_check"] and row["scenario_check"] for row in report["rows"])
    report["finished_at"] = datetime.now().astimezone().isoformat()
    save()
    if not report["passed"]:
        raise RuntimeError("有核验未通过；实际事件、日志与结果已保留")


if __name__ == "__main__":
    main()
