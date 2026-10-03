"""本地Qwen路由/原生批次与真实论文工具串并行核验，不冒充正式路由准确率实验。"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from threading import Lock
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import BaseMessage, HumanMessage

from src.agent.react_loop import _model_request, run_react
from src.agent.router import execute_calls
from src.agent.tools import AVAILABLE_TOOLS, execute_tool
from src.data_loader import batch_import, create_import_tasks
from src.generation.rag_pipeline import urlopen
from src.utils.config import load_config


def serial(value):
    """保留原始工具数据；仅把Message转为可写入JSON的结构。"""
    if isinstance(value, BaseMessage):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serial(item) for item in value]
    return value


def tokens(events):
    """整批Action事件只在第一条计一次；工具内部的实际模型调用另计，未知不补零。"""
    usage = [event["usage"] for event in events if "usage" in event]
    usage += [event["result"]["usage"] for event in events if event["type"] == "tool_result"
              and isinstance(event.get("result"), dict) and "usage" in event["result"]]
    return {key: sum(item[key] for item in usage) if all(type(item.get(key)) is int for item in usage) else None
            for key in ("prompt_eval_count", "eval_count")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("输出已存在，请改用新文件名保留原始实验")
    config = deepcopy(load_config())
    report = {"started_at": datetime.now().astimezone().isoformat(), "config": deepcopy(config),
              "ollama_num_parallel": 1, "rows": [], "validation_complete": False}

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def agent_case(name, question, *, routed=True, expected_calls=1):
        row = {"name": name, "question": question, "events": []}
        report["rows"].append(row)
        started = perf_counter()
        with ExitStack() as stack:
            if not routed:
                stack.enter_context(patch("src.agent.react_loop.route_question", return_value=None))
            for event in run_react(question):
                row["events"].append(serial(event))
                save()
                print(name, event["type"], event.get("name", ""), event.get("tool_name", ""), flush=True)
        events, done = row["events"], row["events"][-1]
        row.update(wall_seconds=perf_counter() - started, usage=tokens(events),
                   agent_model_calls=sum(event["type"] in ("thought", "observation") and bool(event.get("model")) or
                                         event["type"] == "tool_call" and "message" in event for event in events))
        results = [event for event in events if event["type"] == "tool_result"]
        row["passed"] = (done["task_complete"] and done["stop_reason"] == "task_complete"
                         and len(results) == expected_calls and all(event["status"] == "success" for event in results))
        if expected_calls == 2:
            # 验证两调用在同一轮真实并行；最终回答可另经一轮规划，不要求强制完成。
            row["passed"] &= len({event["iteration"] for event in results}) == 1 and all(event["execution_mode"] == "parallel" for event in results)
        save()
        print(name, row["passed"], row["usage"], row["wall_seconds"], done["full_response"], flush=True)

    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        config["paths"]["raw_documents"] = str(Path(directory) / "raw")
        config["paths"]["logs"] = str(Path(directory) / "logs")
        for module in ("src.agent.tools", "src.agent.react_loop", "src.agent.router", "src.generation.rag_pipeline"):
            stack.enter_context(patch(module + ".load_config", return_value=config))
        tasks = create_import_tasks([(name, (ROOT / "data/raw/embedding_papers" / name).read_bytes())
                                     for name in ("attention.pdf", "vit.pdf")])
        list(batch_import(tasks, config["paths"]["raw_documents"]))
        if any(task["status"] != "success" for task in tasks):
            raise RuntimeError(str([(task["name"], task["error"]) for task in tasks]))
        ids = [hashlib.sha256(task["data"]).hexdigest() for task in tasks]
        with urlopen(_model_request([HumanMessage(content="只回答OK。")]), timeout=300) as response:
            report["warmup"] = json.load(response)
        save()  # 预热独立记录，不混入模型调用/延迟比较。
        question = "请返回当前系统时间和时区，只调用current_time一次。"
        agent_case("plain_react_time", question, routed=False)
        agent_case("routed_time", question)
        agent_case("general_concept", "什么是深度学习？", expected_calls=0)
        agent_case("independent_different_tools", "请同时完成两项独立任务：用current_time返回系统时间；"
                   "用keyword_extract提取这句话的关键词：Transformer用于机器翻译，ViT用于图像分类。", expected_calls=2)
        agent_case("independent_papers", f"请同时用paper_metadata分别读取两篇论文的标题和作者，ID为{ids[0]}和{ids[1]}。", expected_calls=2)
        calls = [{"name": "paper_metadata", "args": {"doc_id": doc_id}, "call_id": f"paper-{index}"}
                 for index, doc_id in enumerate(ids)]
        for mode in ("serial", "parallel"):
            row = {"name": "paper_metadata_" + mode, "intervals": [], "events": []}
            report["rows"].append(row)
            lock, started = Lock(), perf_counter()

            def observed_call(name, arguments, tools, call_id):
                """仅记录执行区间，实际调用原执行器；不替换工具或模型返回值。"""
                begin = perf_counter() - started
                event = execute_tool(name, arguments, tools, call_id)
                end = perf_counter() - started
                with lock:
                    row["intervals"].append({"call_id": call_id, "start_seconds": begin, "end_seconds": end})
                return event

            with patch("src.agent.router.execute_tool", side_effect=observed_call):
                for event in execute_calls(calls, AVAILABLE_TOOLS, parallel=mode == "parallel"):
                    row["events"].append(serial(event))
                    save()
            points = sorted([(interval["start_seconds"], 1) for interval in row["intervals"]]
                            + [(interval["end_seconds"], -1) for interval in row["intervals"]])
            active, maximum = 0, 0
            for _, delta in points:
                active += delta
                maximum = max(maximum, active)
            row.update(wall_seconds=perf_counter() - started, max_overlap=maximum, usage=tokens(row["events"]),
                       passed=len(row["events"]) == 2 and all(event["status"] == "success" for event in row["events"])
                       and maximum == (2 if mode == "parallel" else 1))
            save()
            print(row["name"], row["passed"], row["wall_seconds"], row["max_overlap"], row["usage"], flush=True)
    report.update(validation_complete=True, finished_at=datetime.now().astimezone().isoformat())
    save()
    assert all(row["passed"] for row in report["rows"]), "存在实测失败，已保留原始轨迹"


if __name__ == "__main__":
    main()
