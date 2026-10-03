"""八工具下的独立并行专项复测；模型/工具实际执行，不覆盖首次串行结果。"""

import argparse
from datetime import datetime
import json
from importlib import import_module
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

serial = import_module("reports.5_3_3 Agent决策优化.verify_error_recovery").serial
from src.agent.react_loop import run_react
from src.agent.tools import AVAILABLE_TOOLS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError("请换用新报告路径，保留已有实测")
    question = "请同时完成两项独立任务：用current_time返回当前系统时间；用keyword_extract提取文本关键词：Transformer用于机器翻译，ViT用于图像分类。"
    report = {"started_at": datetime.now().astimezone().isoformat(), "question": question,
              "registered_local_tools": [tool.name for tool in AVAILABLE_TOOLS], "events": [], "passed": False}
    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for event in run_react(question):
        report["events"].append(serial(event))
        save()
        print(event["type"], event.get("name", ""), flush=True)
    results = [event for event in report["events"] if event["type"] == "tool_result"]
    report["passed"] = report["events"][-1]["task_complete"] and len(results) == 2 and {
        event["name"] for event in results} == {"current_time", "keyword_extract"} and len({
        event["iteration"] for event in results}) == 1 and all(event["status"] == "success" and
        event["execution_mode"] == "parallel" for event in results)
    report["finished_at"] = datetime.now().astimezone().isoformat()
    save()
    if not report["passed"]:
        raise RuntimeError("独立并行专项仍未通过，真实轨迹已保存")


if __name__ == "__main__":
    main()
