"""模块三完整性联调：真实论文/索引/本机模型，缺项与失败分别记录，不冒充全部验收通过。"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from langchain_core.tools import tool
from reports.verify_error_recovery import serial
from src.agent.memory import MemoryManager, count_memory_tokens, run_session
from src.agent.react_loop import run_react
from src.agent.router import execute_calls
from src.agent.tools import AVAILABLE_TOOLS, execute_tool, get_available_tools, web_search
from src.data_loader import create_import_tasks
from src.retrieval.vector_store import VectorStore, batch_build_index
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError("请使用新路径，不覆盖旧验证记录")
    config = deepcopy(load_config())
    expected = ["knowledge_base_search", "paper_metadata", "paper_compare", "keyword_extract",
                "paper_summary", "current_time", "calculator", "paper_list"]
    registered = [item.name for item in AVAILABLE_TOOLS]
    report = {"started_at": datetime.now().astimezone().isoformat(), "rows": [],
              "registered_local_tools": registered, "missing_local_tools": [name for name in expected if name not in registered],
              "scope": "两篇完整公开AI论文、本地M3E/BGE/Qwen、临时Chroma/SQLite。仅隔离配置路径；"
                       "故障工具和长历史为明确构造的功能样例，不计为生产工具或独立质量评测。",
              "optional_search": "保持默认禁用，只核验开关拒绝，不发出外网请求",
              "validation_complete": False, "module_complete": False}

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def tool_case(name, tool_name, arguments, check):
        row = {"name": name, "kind": "real_tool", "arguments": arguments}
        report["rows"].append(row)
        save()
        event = execute_tool(tool_name, arguments, get_available_tools())
        row.update(event=serial(event), passed=event["status"] == "success" and bool(check(event["result"])))
        save()
        print(name, row["passed"], event.get("error", ""), flush=True)

    def agent_case(name, stream, check):
        row = {"name": name, "kind": "real_agent", "events": []}
        report["rows"].append(row)
        save()
        started = perf_counter()
        try:
            for event in stream:
                row["events"].append(serial(event))
                save()
                print(name, event["type"], event.get("stop_reason", event.get("name", "")), flush=True)
            row["passed"] = bool(row["events"] and check(row["events"][-1], row["events"]))
        except Exception as error:
            row.update(passed=False, error=f"{type(error).__name__}: {error}")
        row["elapsed_seconds"] = perf_counter() - started
        save()
        return row

    save()
    with tempfile.TemporaryDirectory(prefix="rag-module-three-") as directory, ExitStack() as stack:
        for key, name in (("raw_documents", "raw"), ("vector_index", "index"), ("logs", "logs"), ("session_db", "memory.sqlite3")):
            config["paths"][key] = str(Path(directory) / name)
        for module in ("src.agent.tools", "src.agent.react_loop", "src.agent.router", "src.agent.memory",
                       "src.retrieval.vector_store", "src.retrieval.hybrid_retriever", "src.retrieval.reranker",
                       "src.generation.rag_pipeline", "src.utils.logger", "src.chunking"):
            stack.enter_context(patch(module + ".load_config", return_value=config))
        report["config"] = deepcopy(config)
        tasks = create_import_tasks([(name, (ROOT / "data/raw/embedding_papers" / name).read_bytes())
                                     for name in ("attention.pdf", "vit.pdf")])
        store = VectorStore()
        for progress in batch_build_index(tasks, config["paths"]["raw_documents"], vector_store=store):
            if progress["completed"]:
                print("入库", progress, flush=True)
        report["ingestion"] = [{"file": task["name"], "status": task["status"], "indexed": task.get("indexed"),
                                "chunks": task.get("chunk_count"), "error": task.get("error")} for task in tasks]
        save()
        if not all(task["status"] == "success" and task.get("indexed") for task in tasks):
            raise RuntimeError("论文入库失败，已保存真实状态")
        ids = [task["documents"][0].metadata["doc_id"] for task in tasks]
        report["paper_ids"] = ids
        for index, title, year in ((0, "Attention Is All You Need", 2017), (1, "AN IMAGE IS WORTH", 2021)):
            tool_case("metadata_" + tasks[index]["name"], "paper_metadata", {"doc_id": ids[index]},
                      lambda result, title=title, year=year: title.lower() in (result.get("title") or "").lower()
                      and result.get("year") == year and bool(result.get("authors")) and bool(result.get("abstract"))
                      and bool(result.get("evidence")))
        tool_case("question_keywords", "keyword_extract", {"text": "Transformer用于机器翻译，ViT用于图像分类。"},
                  lambda result: bool(result.get("keywords")) and bool(result.get("evidence")))
        tool_case("paper_summary", "paper_summary", {"doc_id": ids[0]},
                  lambda result: result.get("status") == "answered" and len(result.get("sections", {})) == 4
                  and bool(result.get("citations")) and all(ref["metadata"]["doc_id"] == ids[0] for ref in result["citations"]))
        tool_case("paper_compare", "paper_compare", {"paper_a_id": ids[0], "paper_b_id": ids[1]},
                  lambda result: result.get("status") == "answered"
                  and {ref["metadata"]["doc_id"] for ref in result.get("citations", [])} == set(ids)
                  and "Transformer" in result.get("answer", "") and "WMT" in result.get("answer", ""))
        before = datetime.now().astimezone() - timedelta(seconds=1)
        tool_case("current_time", "current_time", {}, lambda result:
                  before <= datetime.fromisoformat(result["system_time"]) <= datetime.now().astimezone()
                  and datetime.fromisoformat(result["system_time"]).utcoffset() is not None)

        memory = MemoryManager()
        session = memory.create_session("alice")
        question = (f"请调用knowledge_base_search查询论文ID {ids[0]}：Transformer在WMT 2014英德翻译任务的BLEU分数是多少？"
                    "回答保留实际工具给出的文档名和页码引用。")
        def rag_check(done, events):
            results = [event["result"] for event in events if event["type"] == "tool_result"
                       and event["name"] == "knowledge_base_search" and event["status"] == "success"]
            return done["task_complete"] and "28.4" in done["full_response"] and "attention.pdf" in done["full_response"] and any(
                result.get("status") == "answered" and "28.4" in result.get("answer", "") and result.get("citations") for result in results)
        agent_case("session_rag_citations", run_session(question, "alice", session, memory=memory), rag_check)
        agent_case("session_followup", run_session("刚才给出的BLEU分数是多少？请只给数字。", "alice", session, tools=[], memory=memory),
                   lambda done, events: done["task_complete"] and "28.4" in done["full_response"])
        agent_case("independent_parallel_tools", run_react("请同时完成两项独立任务：用current_time返回当前系统时间；"
                   "用keyword_extract提取文本关键词：Transformer用于机器翻译，ViT用于图像分类。"),
                   lambda done, events: done["task_complete"] and len([e for e in events if e["type"] == "tool_result"
                   and e["status"] == "success" and e.get("execution_mode") == "parallel"]) == 2)

        # 开发工具仅用于验证依赖执行/终止，绝不计入正式工具数量。
        @tool
        def multiply(a: float, b: float) -> float:
            """计算两个数的乘积。"""
            return a * b
        @tool
        def add(a: float, b: float) -> float:
            """计算两个数的和。"""
            return a + b
        calculation = "请分两步用工具执行：先multiply计算3乘4，再add将所得乘积加5，最后报告结果。"
        agent_case("dependent_two_steps", run_react(calculation, [multiply, add]), lambda done, events:
                   done["task_complete"] and [o["result"] for o in done["context"]["observations"]] == [12, 17]
                   and "17" in done["full_response"])
        original_limit = config["agent"]["max_iterations"]
        config["agent"]["max_iterations"] = 1
        agent_case("iteration_limit", run_react(calculation, [multiply, add]), lambda done, events:
                   not done["task_complete"] and done["stop_reason"] == "max_iterations" and done["iterations"] == 1)
        config["agent"]["max_iterations"] = original_limit

        attempts = []
        @tool
        def retry_clock() -> dict:
            """读取系统时间；此测试工具首次主动注入已结束的超时。"""
            attempts.append(datetime.now().astimezone().isoformat())
            if len(attempts) == 1:
                raise TimeoutError("模块完整性验证主动注入的超时，函数已结束")
            return {"system_time": datetime.now().astimezone().isoformat()}
        results = list(execute_calls([{"name": "retry_clock", "args": {}, "call_id": "retry-test"}], [retry_clock]))
        report["rows"].append({"name": "injected_timeout_retry", "kind": "injected_fault", "events": serial(results),
                               "passed": len(attempts) == 2 and results[0]["status"] == "success"
                               and [a["status"] for a in results[0]["attempts"]] == ["error", "success"]})

        long_session = memory.create_session("alice")
        bob = memory.create_session("bob")
        memory.append_turn("bob", bob, "BOB_PRIVATE", "Bob私有记录")
        code = "EXP_" + uuid4().hex[:10].upper()
        memory.append_turn("alice", long_session, f"我的当前实验代号是{code}。", f"已记录用户的实验代号{code}。")
        for index in range(1, 12):
            memory.append_turn("alice", long_session, f"第{index}轮是人工构造的长对话功能样例。", "保留中文回答和核验论文来源的要求。")
        row = agent_case("long_memory_summary_followup", run_session("我的当前实验代号是什么？只回答代号。", "alice", long_session, tools=[], memory=memory),
                         lambda done, events: done["task_complete"] and code in done["full_response"]
                         and done["context"].get("memory_summary", {}).get("summarized_turns") == 8
                         and count_memory_tokens(done["context"]["history"], done["context"].get("summary", "")) <= 2000
                         and "BOB_PRIVATE" not in json.dumps(done["context"], ensure_ascii=False))
        row["archive_turns_after"] = len(memory.get_messages("alice", long_session)) // 2
        row["passed"] &= row["archive_turns_after"] == 13
        memory.clear_session("alice", long_session)
        cleared = MemoryManager(memory.db_path).get_context("alice", long_session)
        report["rows"].append({"name": "summary_clear_and_isolation", "kind": "real_sqlite", "passed":
                               cleared["history"] == [] and "summary" not in cleared and len(memory.get_messages("bob", bob)) == 2})
        denied = execute_tool("web_search", {"query": "测试开关拒绝，不应联网"}, [web_search])
        report["rows"].append({"name": "disabled_search", "kind": "offline_gate", "event": serial(denied),
                               "passed": denied["status"] == "error" and web_search not in get_available_tools()})
        from src.utils.logger import read_rag_requests
        logs, invalid = read_rag_requests()
        report["rag_logs"] = logs
        report["invalid_log_lines"] = invalid
        report["live_checks_passed"] = all(row["passed"] for row in report["rows"])
        report["module_complete"] = report["live_checks_passed"] and not report["missing_local_tools"]
        report["validation_complete"] = True  # 表示核验执行完毕，不代表功能/模块全通过。
        report["finished_at"] = datetime.now().astimezone().isoformat()
        save()
    if not report["live_checks_passed"]:
        raise RuntimeError("联调有失败项，已保留全部真实结果")


if __name__ == "__main__":
    main()
