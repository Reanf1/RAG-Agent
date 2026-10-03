"""只读核验并发测试已落盘的事件、SQLite和日志；不重新生成答案或修改失败记录。"""

import argparse
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reports.verify_session_isolation import serial
from src.agent.memory import MemoryManager, run_session
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config


def logs(directory, prefix):
    """原始JSONL逐行解析；损坏行直接失败，不跳过错误来拼成功报告。"""
    return [json.loads(line) for path in directory.glob(prefix + "_*.jsonl")
            for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--ui-root", type=Path, required=True)
    parser.add_argument("--ui-identities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("请使用新报告文件，不覆盖原始失败证据")
    source = json.loads(args.backend.read_text())
    root = Path(source["config"]["paths"]["session_db"]).parent
    memory = MemoryManager(root / "memory.sqlite3")
    report = {"checked_at": datetime.now().astimezone().isoformat(), "source": str(args.backend),
              "checks": {}, "backend_groups": source["groups"], "ui_rows": []}
    checks = report["checks"]
    checks["application_overlap"] = all(g["peak_active_requests"] == len(g["rows"]) for g in source["groups"])
    checks["owned_events_and_history"] = all(all(value for key, value in r["checks"].items() if key != "completed")
        for g in source["groups"] for r in g["rows"])
    rows = [r for g in source["groups"] for r in g["rows"]]
    report["backend_agent_logs"] = logs(root / "logs", "agent")
    finals = [r for r in report["backend_agent_logs"] if r["event"] == "done"]
    checks["five_owned_final_logs"] = len(finals) == 5 and all(sum(
        log["request_id"] == row["events"][-1]["request_id"] and log["session_id"] == row["session_id"] and
        log["user_id"] == row["user_id"] for log in finals) == 1 for row in rows)
    checks["persisted_backend_turns"] = all(serial(memory.get_messages(r["user_id"], r["session_id"])) ==
                                             r["history"] for r in rows)
    try:
        list(run_session("尝试越权读取", rows[1]["user_id"], rows[0]["session_id"], memory=memory))
    except PermissionError:
        checks["foreign_access_rejected"] = True
    else:
        checks["foreign_access_rejected"] = False
    report["rag_logs"] = logs(root / "logs", "rag")
    checks["two_rag_requests_returned_real_chunks"] = len(report["rag_logs"]) == 2 and all(
        r["status"] in {"completed", "incomplete"} and r["retrieval"]["status"] == "success" and
        len(r["retrieval"]["documents"]) == 5 for r in report["rag_logs"])
    checks["shared_index_unchanged"] = serial(VectorStore(root / "index").list_chunks()) == serial(
        VectorStore(Path(source["source"]) / "index").list_chunks())
    baseline = source["baseline"]
    checks["business_paths_unchanged"] = baseline == {
        "config_sha256": sha256((ROOT / "config.yaml").read_bytes()).hexdigest(),
        "default_session_exists": (ROOT / load_config()["paths"]["session_db"]).exists(),
        "index_ids": sorted(d.metadata["chunk_id"] for d in VectorStore().list_chunks())}
    ui = MemoryManager(args.ui_root / "memory.sqlite3")
    report["ui_agent_logs"] = logs(args.ui_root / "logs", "agent")
    ui_finals = [r for r in report["ui_agent_logs"] if r["event"] == "done"]
    identities = json.loads(args.ui_identities.read_text())
    for identity in identities:
        messages = serial(ui.get_messages(identity["user_id"], identity["session_id"]))
        own_logs = [r for r in report["ui_agent_logs"] if r["user_id"] == identity["user_id"]]
        final = next(r for r in ui_finals if r["session_id"] == identity["session_id"])
        started = datetime.fromisoformat(final["timestamp"]) - timedelta(seconds=final["metrics"]["response_seconds"])
        report["ui_rows"].append({**identity, "history": messages, "final_log": final,
                                  "started_at_estimated": started.isoformat(), "logs": own_logs})
    checks["ui_owned_answers_and_histories"] = len(ui_finals) == 2 and all(
        len(r["history"]) == 4 and r["history"][-1]["content"] == r["code"] and r["final_log"]["task_complete"] and
        all(other["code"] not in json.dumps(r, ensure_ascii=False) for other in identities if other["user_id"] != r["user_id"])
        for r in report["ui_rows"])
    # 用完整响应耗时反推开始时刻，误差仅包含最后一条日志的落盘开销；间隔秒数另保存。
    overlap = min(datetime.fromisoformat(r["final_log"]["timestamp"]) for r in report["ui_rows"]) - max(
        datetime.fromisoformat(r["started_at_estimated"]) for r in report["ui_rows"])
    report["ui_overlap_seconds_estimated"] = overlap.total_seconds()
    checks["ui_requests_overlap"] = overlap.total_seconds() > 1
    report["task_completed"] = sum(r["events"][-1]["task_complete"] for r in rows) + sum(r["task_complete"] for r in ui_finals)
    report["request_count"] = len(rows) + len(ui_finals)
    report["concurrency_checks_passed"] = all(checks.values())
    report["all_answer_tasks_completed"] = report["task_completed"] == report["request_count"]
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checks": checks, "task_completed": report["task_completed"], "request_count": report["request_count"],
                      "concurrency_checks_passed": report["concurrency_checks_passed"],
                      "all_answer_tasks_completed": report["all_answer_tasks_completed"]}, ensure_ascii=False, indent=2))
    if not report["concurrency_checks_passed"]:
        raise RuntimeError("并发机制核验有失败；完整证据已保存")


if __name__ == "__main__":
    main()
