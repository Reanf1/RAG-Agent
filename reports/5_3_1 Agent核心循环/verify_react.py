"""真实本地模型验证Observation与循环终止；不模拟模型或工具结果。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage

from src.agent.react_loop import run_react
from src.generation.rag_pipeline import urlopen
from src.utils.config import load_config
from src.utils.logger import request_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/5_3_1 Agent核心循环/ReAct循环验证结果_20261002.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("请指定不存在的新路径，保留历史实测记录")
    executions = []

    @tool
    def multiply(a: float, b: float) -> float:
        """计算两个数的乘积。"""
        executions.append({"name": "multiply", "args": {"a": a, "b": b}})
        return a * b

    @tool
    def add(a: float, b: float) -> float:
        """计算两个数的和。"""
        executions.append({"name": "add", "args": {"a": a, "b": b}})
        return a + b

    @tool
    def divide(a: float, b: float) -> float:
        """实际执行a除以b，除零时抛出异常。"""
        executions.append({"name": "divide", "args": {"a": a, "b": b}})
        return a / b

    tools = [multiply, add, divide]
    calculation = "请按两步执行：先用multiply计算3乘4，然后用add把得到的乘积加5，最后报告结果。每一步都必须使用对应工具。"
    cases = [("two_steps", calculation, tools, 8, "task_complete"),
             ("concept", "什么是过拟合？", [], 8, "task_complete"),
             ("missing_paper", "请告诉我未上传的论文A在ImageNet上的准确率，并引用该论文页码。", [], 8, "incomplete"),
             ("tool_error", "请调用divide工具实际计算1除以0，依据真实结果报告。", tools, 8, "incomplete"),
             ("iteration_limit", calculation, tools, 1, "max_iterations")]
    config = load_config()
    report = {"started_at": request_time(), "config": config,
              "server": json.load(urlopen(config["llm"]["base_url"] + "/api/version", timeout=10)),
              "scope": "真实Qwen与三个本地开发工具验证核心循环，不计为八个生产工具或系统质量评测。"
                       "仅每例agent.max_iterations通过局部配置替换指定，模型HTTP和工具均真实执行。",
              "validation_complete": False, "rows": []}
    for name, question, available, limit, expected in cases:
        before = len(executions)
        local_config = deepcopy(config)
        local_config["agent"]["max_iterations"] = limit
        row = {"name": name, "question": question, "max_iterations": limit,
               "expected_stop_reason": expected, "events": []}
        report["rows"].append(row)
        # 仅更改迭代上限用于边界实测；llm配置、请求与执行函数不作mock。
        with patch("src.agent.react_loop.load_config", return_value=local_config):
            for event in run_react(question, available):
                row["events"].append({key: value.model_dump() if isinstance(value, BaseMessage) else value
                                      for key, value in event.items()})
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(name, event["type"], event.get("decision", event.get("stop_reason", "")), flush=True)
        row["executions"] = executions[before:]
        done = row["events"][-1]
        row["passed"] = (done["type"] == "done" and done["stop_reason"] == expected
                         and done["task_complete"] == (expected == "task_complete"))
        if name == "two_steps":
            row["passed"] &= (done["iterations"] == 2
                              and [o["result"] for o in done["context"]["observations"]] == [12, 17]
                              and "17" in done["full_response"])
        elif name in ("concept", "missing_paper"):
            row["passed"] &= not row["executions"]
        elif name == "tool_error":
            row["passed"] &= (len(row["executions"]) == 1
                              and "ZeroDivisionError" in done["context"]["observations"][0]["error"])
        elif name == "iteration_limit":
            row["passed"] &= (done["iterations"] == 1 and len(row["executions"]) == 1)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["validation_complete"], report["finished_at"] = True, request_time()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert all(row["passed"] for row in report["rows"]), "存在实测失败，请检查保留的原始轨迹"


if __name__ == "__main__":
    main()
