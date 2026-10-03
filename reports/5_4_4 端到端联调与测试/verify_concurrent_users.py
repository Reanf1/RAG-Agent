"""真实本地模型多用户并发核验；共享论文索引，独立会话，保留所有事件与失败。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import json
from hashlib import sha256
import os
from importlib import import_module
from pathlib import Path
import shutil
import sys
from threading import Barrier, Lock, local
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
serial = import_module("reports.5_3_4 多轮对话记忆管理.verify_session_isolation").serial
from src.agent import react_loop
from src.agent.memory import MemoryManager, run_session
from src.retrieval.reranker import get_reranker
from src.retrieval.vector_store import VectorStore, get_embeddings
from src.utils.config import load_config
from src.utils.logger import read_rag_requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="已验证的真实论文raw/index目录")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory, output = args.root.resolve(), args.output.resolve()
    if directory.exists() or output.exists():
        raise FileExistsError("请使用新目录和新报告路径，不覆盖历史证据")
    directory.mkdir(parents=True)
    # 复用上一项完整流程的真实94块索引；本测试不重复冒充一次上传/入库实验。
    for folder in ("raw", "index"):
        shutil.copytree(args.source / folder, directory / folder)
    config = deepcopy(load_config())
    baseline = {"config_sha256": sha256((ROOT / "config.yaml").read_bytes()).hexdigest(),
                "default_session_exists": (ROOT / config["paths"]["session_db"]).exists(),
                "index_ids": sorted(d.metadata["chunk_id"] for d in VectorStore().list_chunks())}
    for key, folder in (("raw_documents", "raw"), ("vector_index", "index"), ("logs", "logs")):
        config["paths"][key] = str(directory / folder)
    config["paths"]["session_db"] = str(directory / "memory.sqlite3")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    report = {"started_at": datetime.now().astimezone().isoformat(), "config": config,
              "source": str(args.source.resolve()), "baseline": baseline, "groups": [], "passed": False}
    lock, owner = Lock(), local()
    def save():
        # 调用方持锁；即使模型失败也逐事件保存原始证据。
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    real_http = react_loop.urlopen
    def record_http(request, *params, **kwargs):
        with lock:
            owner.row["model_requests"].append(json.loads(request.data))
            save()
        return real_http(request, *params, **kwargs)  # 实际网络响应原样消费，不构造答案。
    # 路径替换在所有线程启动前统一安装，结束后统一还原，不在各线程内嵌套patch。
    with ExitStack() as stack:
        for module in ("src.utils.config", "src.agent.memory", "src.utils.logger", "src.agent.react_loop",
                       "src.agent.tools", "src.agent.router", "src.retrieval.vector_store",
                       "src.retrieval.hybrid_retriever", "src.retrieval.reranker",
                       "src.generation.rag_pipeline", "src.generation.cache", "src.chunking"):
            stack.enter_context(patch(module + ".load_config", return_value=config))
        stack.enter_context(patch("src.agent.react_loop.urlopen", side_effect=record_http))
        memory = MemoryManager()
        ids_before = sorted(d.metadata["chunk_id"] for d in VectorStore().list_chunks())
        started = perf_counter()
        get_embeddings()
        get_reranker()
        report["embedding_reranker_load_seconds"] = perf_counter() - started
        for name, count in (("三用户历史追问", 3), ("两用户论文检索", 2)):
            group = {"name": name, "rows": []}
            report["groups"].append(group)
            for index in range(count):
                user = uuid4().hex
                row = {"user_id": user, "session_id": memory.create_session(user),
                       "code": "TEST_" + uuid4().hex[:12].upper(), "events": [], "model_requests": []}
                if count == 3:
                    # 明确构造的历史样例用于隔离验证，不是论文资料或模型生成记录。
                    memory.append_turn(user, row["session_id"], f"本会话实验代号是{row['code']}。", "已记录。")
                    row["question"] = "我刚才告诉你的实验代号是什么？请只回答代号。"
                else:
                    row["question"] = [
                        "请根据已上传论文 attention.pdf 的原文，说明 Transformer 的编码器有多少层、输出维度 d_model 是多少。请给出文档名和页码引用。",
                        "请根据已上传论文 attention.pdf 的原文，说明多头注意力使用多少个头 h，每个头的维度 d_k 是多少。请给出文档名和页码引用。",
                    ][index]
                group["rows"].append(row)
            barrier = Barrier(count)
            def work(row):
                owner.row = row
                barrier.wait(timeout=10)
                row["start_seconds"] = perf_counter()
                try:
                    for event in run_session(row["question"], row["user_id"], row["session_id"],
                                             tools=[] if count == 3 else None, memory=memory):
                        with lock:
                            row["events"].append(serial(event))
                            save()
                        print(name, row["user_id"][:8], event["type"], flush=True)
                except Exception as error:
                    row["exception"] = f"{type(error).__name__}: {error}"
                finally:
                    with lock:
                        row["end_seconds"] = perf_counter()
                        row["elapsed_seconds"] = row["end_seconds"] - row["start_seconds"]
                        save()
            with ThreadPoolExecutor(max_workers=count) as pool:
                list(pool.map(work, group["rows"]))
            for row in group["rows"]:
                done = row["events"][-1] if row["events"] else {}
                history = serial(memory.get_messages(row["user_id"], row["session_id"]))
                row["history"] = history
                foreign_codes = [r["code"] for r in group["rows"] if r is not row]
                row["checks"] = {
                    "completed": done.get("type") == "done" and done.get("task_complete") is True,
                    "event_ownership": bool(row["events"]) and all(e["user_id"] == row["user_id"] and
                        e["session_id"] == row["session_id"] for e in row["events"]),
                    "history_count": len(history) == (4 if count == 3 else 2),
                    "no_foreign_history": all(code not in json.dumps(row, ensure_ascii=False) for code in foreign_codes),
                    "saved_answer": bool(history) and history[-1]["content"] == done.get("full_response"),
                    "own_code_or_rag_tool": row["code"] in done.get("full_response", "") if count == 3 else
                        any(e["type"] == "tool_result" and e["name"] == "knowledge_base_search" and
                            e["status"] == "success" for e in row["events"]),
                    "no_log_error": not any(e.get("log_error") for e in row["events"]),
                }
                row["passed"] = all(row["checks"].values()) and "exception" not in row
            intervals = sorted([(r["start_seconds"], 1) for r in group["rows"]] +
                               [(r["end_seconds"], -1) for r in group["rows"]])
            active, peak = 0, 0
            for _, change in intervals:
                active += change
                peak = max(peak, active)
            group.update(peak_active_requests=peak,
                         wall_seconds=max(r["end_seconds"] for r in group["rows"]) - min(r["start_seconds"] for r in group["rows"]),
                         mean_seconds=sum(r["elapsed_seconds"] for r in group["rows"]) / count,
                         passed=peak == count and all(r["passed"] for r in group["rows"]))
            save()
        rows = [r for g in report["groups"] for r in g["rows"]]
        try:
            list(run_session("尝试越权读取", rows[1]["user_id"], rows[0]["session_id"], memory=memory))
        except PermissionError:
            report["foreign_access_rejected"] = True
        else:
            report["foreign_access_rejected"] = False
        report["rag_logs"], invalid = read_rag_requests()
        report["index_unchanged"] = ids_before == sorted(d.metadata["chunk_id"] for d in VectorStore().list_chunks())
        report["log_checks"] = {"valid_rag_lines": invalid == 0, "two_terminal_rag_requests":
                               len(report["rag_logs"]) == 2 and all(r["status"] in {"completed", "incomplete"} and
                                   r["retrieval"]["status"] == "success" for r in report["rag_logs"])}
        report["agent_logs"] = [json.loads(line) for path in (directory / "logs").glob("agent_*.jsonl")
                                for line in path.read_text().splitlines() if line.strip()]
        finals = [r for r in report["agent_logs"] if r["event"] == "done"]
        report["log_checks"]["five_owned_final_logs"] = len(finals) == 5 and all(
            sum(log["request_id"] == row["events"][-1]["request_id"] and log["user_id"] == row["user_id"] and
                log["session_id"] == row["session_id"] for log in finals) == 1 for row in rows if row["events"])
    report["baseline_unchanged"] = baseline == {
        "config_sha256": sha256((ROOT / "config.yaml").read_bytes()).hexdigest(),
        "default_session_exists": (ROOT / load_config()["paths"]["session_db"]).exists(),
        "index_ids": sorted(d.metadata["chunk_id"] for d in VectorStore().list_chunks())}
    report["finished_at"] = datetime.now().astimezone().isoformat()
    report["passed"] = all(g["passed"] for g in report["groups"]) and report["foreign_access_rejected"] and \
        report["index_unchanged"] and report["baseline_unchanged"] and all(report["log_checks"].values())
    save()
    if not report["passed"]:
        raise RuntimeError("并发核验存在失败；完整请求、事件和历史已保存")


if __name__ == "__main__":
    main()
