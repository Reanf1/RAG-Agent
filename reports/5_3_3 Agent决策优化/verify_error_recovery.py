"""错误恢复功能核验：真实本地Qwen与工具执行，故障显式注入，不算科研质量评测。"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool

from src.agent.react_loop import run_react
from src.utils.config import load_config


def serial(value):
    """保留事件和实际错误，消息转换为JSON可保存结构。"""
    if isinstance(value, BaseMessage):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serial(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("请换一个新输出路径，保留原始核验记录")
    config = deepcopy(load_config())
    report = {"started_at": datetime.now().astimezone().isoformat(), "config": config,
              "faults_injected": True, "rows": [], "validation_complete": False}

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for mode in ("execution_failure", "timeout_once", "timeout_exhausted"):
        calls, events = [], []
        @tool
        def primary_clock() -> dict:
            """首选系统时钟查询。获取本机当前时间；如执行失败，使用备用时钟工具。"""
            calls.append({"name": "primary_clock", "at": datetime.now().astimezone().isoformat()})
            if mode == "execution_failure":
                raise RuntimeError("验证主动注入的主时钟故障，必须从备用时钟获取真实时间")
            if mode == "timeout_exhausted" or mode == "timeout_once" and len(calls) == 1:
                raise TimeoutError("验证主动注入的已结束超时；函数本次执行已经退出")
            return {"iso_time": datetime.now().astimezone().isoformat()}
        @tool
        def backup_clock() -> dict:
            """备用系统时钟查询。主时钟执行失败时，读取本机当前时间完成同一个任务。"""
            calls.append({"name": "backup_clock", "at": datetime.now().astimezone().isoformat()})
            return {"iso_time": datetime.now().astimezone().isoformat()}
        row = {"name": mode, "events": events, "actual_calls": calls}
        report["rows"].append(row)
        save()
        # 只关闭首轮规则路由以固定主工具优先条件，三个阶段的模型HTTP均为真实调用。
        with patch("src.agent.react_loop.route_question", return_value=None):
            for event in run_react("请先用primary_clock读取系统当前时间；仅在主时钟执行失败时用备用时钟完成同一任务。", [primary_clock, backup_clock]):
                events.append(serial(event))
                save()
                print(mode, event["type"], event.get("name", ""), event.get("tool_name", ""), flush=True)
        done = events[-1]
        names = [call["name"] for call in calls]
        results = [event for event in events if event["type"] == "tool_result"]
        row["passed"] = done["task_complete"] and done["stop_reason"] == "task_complete"
        row["passed"] &= names == (["primary_clock", "backup_clock"] if mode == "execution_failure" else
                                   ["primary_clock", "primary_clock"] if mode == "timeout_once" else
                                   ["primary_clock", "primary_clock", "backup_clock"])
        row["passed"] &= results[-1]["status"] == "success" and "iso_time" in results[-1]["result"]
        if mode != "timeout_once":
            row["passed"] &= results[0]["status"] == "error" and results[0]["call_id"] != results[1]["call_id"]
        else:
            row["passed"] &= [a["status"] for a in results[0]["attempts"]] == ["error", "success"]
        save()
    report["validation_complete"] = all(row["passed"] for row in report["rows"])
    report["finished_at"] = datetime.now().astimezone().isoformat()
    save()
    if not report["validation_complete"]:
        raise RuntimeError("存在未通过的真实模型核验，原始事件已保存")


if __name__ == "__main__":
    main()
