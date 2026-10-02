"""真实本地Thought→Function Calling→工具执行，记录成功、异常与跳过。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from langchain_core.tools import tool

from src.agent.react_loop import act, think
from src.generation.rag_pipeline import urlopen
from src.utils.config import load_config
from src.utils.logger import request_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/Action工具执行验证结果_20261002.json")
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
    def divide(a: float, b: float) -> float:
        """计算a除以b，分母为零时实际抛出异常。"""
        executions.append({"name": "divide", "args": {"a": a, "b": b}})
        return a / b

    @tool
    def current_time() -> str:
        """返回系统当前北京时间，不需要参数。"""
        executions.append({"name": "current_time", "args": {}})
        return request_time()

    tools = [multiply, divide, current_time]
    cases = [("multiply", "请用工具计算3.14乘以2.56。", "success"),
             ("current_time", "请用当前时间工具返回北京时间。", "success"),
             ("divide", "请调用divide工具计算1除以0，实际尝试并记录异常。", "error"),
             (None, "什么是过拟合？", "skipped")]
    config = load_config()
    report = {"started_at": request_time(), "config": config,
              "server": json.load(urlopen(config["llm"]["base_url"] + "/api/version", timeout=10)),
              "models": json.load(urlopen(config["llm"]["base_url"] + "/api/tags", timeout=10)),
              "scope": "真实Qwen单轮Thought/Action与三个实际执行的本地开发样例工具，不计为生产八工具、完整循环、并行或恢复评测。",
              "validation_complete": False, "rows": []}
    for expected_name, question, expected_status in cases:
        before = len(executions)
        thought = think(question, tools)
        events = []
        for event in act(question, thought, tools):
            # 文件只序列化消息快照，运行接口保持真实LangChain消息对象。
            events.append({key: value.model_dump() if key == "message" else value for key, value in event.items()})
        row = {"question": question, "thought": thought, "events": events,
               "executions": executions[before:], "expected_tool": expected_name, "expected_status": expected_status}
        report["rows"].append(row)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if expected_status == "skipped":
            assert thought["next_step"] == "answer" and len(executions) == before
            assert [event["type"] for event in events] == ["action_skipped"]
        else:
            assert thought["tool_name"] == expected_name and len(executions) == before + 1
            assert [event["type"] for event in events] == ["tool_call", "tool_result"]
            assert events[-1]["status"] == expected_status
            assert events[0]["call_id"] == events[-1]["call_id"]
            if expected_name == "multiply":
                assert abs(events[-1]["result"] - 8.0384) < 1e-10
            elif expected_name == "divide":
                assert "ZeroDivisionError" in events[-1]["error"]
        print(expected_name, expected_status, events[-1].get("result"), flush=True)
    report["validation_complete"], report["finished_at"] = True, request_time()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
