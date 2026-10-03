"""只读复核边缘场景的真实输入输出、索引、SQLite摘要及浏览器记录。"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reports.verify_session_isolation import serial
from src.agent.memory import MemoryManager, count_memory_tokens
from src.generation.prompt_template import NO_CONTEXT_TEXT
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--ui-root", type=Path, required=True)
    parser.add_argument("--ui-identities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("使用新输出文件，保留此前失败记录")
    backend = json.loads(args.backend.read_text())
    root = Path(backend["config"]["paths"]["session_db"]).parent
    memory = MemoryManager(root / "memory.sqlite3")
    rows = backend["rows"]
    checks = {"backend_function_checks": all(backend["checks"].values()) and all(
        value for row in rows for key, value in row["checks"].items() if key != "summary_saved")}
    # 摘要通过Context和SQLite判断，业务run_react仅将摘要调用汇入metrics，不单独yield事件。
    with sqlite3.connect(memory.db_path) as connection:
        summaries = {sid: {"content": content, "through_message_id": boundary} for sid, content, boundary in
                     connection.execute("SELECT session_id, content, through_message_id FROM summaries")}
    long = next(row for row in rows if row["name"] == "fifty_turn_history")
    context = long["contexts"][0]
    checks["summary_persisted"] = long["session_id"] in summaries and summaries[long["session_id"]]["content"] == \
        context["summary"] and context["memory_summary"]["summarized_turns"] == 46 and \
        all(call["saved"] for call in context["memory_summary"]["calls"])
    huge = next(row for row in rows if row["name"] == "oversized_single_old_turn")
    checks["huge_turn_did_not_save_fake_summary"] = huge["session_id"] not in summaries and not huge["contexts"][0].get("summary")
    checks["archives_match_real_sqlite"] = all(serial(memory.get_messages("edge-verifier", row["session_id"])) ==
                                               row["history_after"] for row in rows)
    checks["actual_model_packets_finished"] = all(packet.get("response", {}).get("done") is True and
        packet["response"]["done_reason"] == "stop" for row in rows for packet in row["model_packets"])
    chunks = VectorStore(root / "upload-index").list_chunks()
    checks["two_valid_uploads_persisted"] = len(chunks) == 2 and all(
        sha256(Path(d.metadata["source"]).read_bytes()).hexdigest() == d.metadata["doc_id"] and
        Path(d.metadata["source"]).stat().st_size in {20 * 1024 * 1024 - 1, 20 * 1024 * 1024} for d in chunks)
    checks["oversized_files_not_saved"] = not any((root / "upload-raw").rglob("limit_plus_one.pdf")) and \
        not any((root / "upload-raw").rglob("oversized_21MiB.pdf"))
    baseline = backend["baseline"]
    checks["business_paths_unchanged"] = baseline == {
        "config_sha256": sha256((ROOT / "config.yaml").read_bytes()).hexdigest(),
        "default_session_exists": (ROOT / load_config()["paths"]["session_db"]).exists(),
        "index_ids": sorted(d.metadata["chunk_id"] for d in VectorStore().list_chunks())}
    ui = MemoryManager(args.ui_root / "memory.sqlite3")
    identities = json.loads(args.ui_identities.read_text())
    empty, older = [next(i for i in identities if i["label"] == label) for label in ("empty", "long")]
    rag = ui.get_rag_messages(empty["user_id"], empty["session_id"])
    messages = serial(ui.get_messages(older["user_id"], older["session_id"]))
    with sqlite3.connect(ui.db_path) as connection:
        summary, boundary = connection.execute("SELECT content, through_message_id FROM summaries WHERE session_id=?",
                                               (older["session_id"],)).fetchone()
    event = messages[-1]["additional_kwargs"]["event"]
    # 前端页面没有保存全部模型请求，使用已落盘的实际done Context核对摘要来源与窗口。
    c = event["context"]
    checks["ui_empty_stream_fallback"] = len(rag) == 1 and rag[0]["complete"] and rag[0]["generation_mode"] == "empty" and \
        NO_CONTEXT_TEXT in rag[0]["answer"] and not rag[0]["citations"] and not rag[0]["retrieved_documents"]
    checks["ui_old_code_only_in_summary"] = older["expected_code"] in summary and summary == c["summary"] and \
        older["expected_code"] not in json.dumps(c["history"], ensure_ascii=False)
    checks["ui_long_answer_and_archive"] = len(messages) == 26 and messages[-1]["content"].strip() == older["expected_code"] and event["task_complete"]
    checks["ui_window_bounded"] = count_memory_tokens(c["history"], c["summary"]) <= 2000 and c["history_window"]["retained_turns"] == 4
    agent_logs = [json.loads(line) for path in (args.ui_root / "logs").glob("agent_*.jsonl") for line in path.read_text().splitlines()]
    rag_logs = [json.loads(line) for path in (args.ui_root / "logs").glob("rag_*.jsonl") for line in path.read_text().splitlines()]
    checks["ui_single_requests_logged"] = sum(log["event"] == "done" for log in agent_logs) == 1 and len(rag_logs) == 3
    report = {"backend_source": str(args.backend), "backend_raw": backend, "checks": checks, "summaries": summaries,
              "ui": {"identities": identities, "rag_history": rag, "agent_history": messages, "summary": summary,
                     "through_message_id": boundary, "agent_logs": agent_logs, "rag_logs": rag_logs},
              "original_summary_check_was_wrong": long["checks"]["summary_saved"] is False, "passed": all(checks.values())}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checks": checks, "passed": report["passed"]}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise RuntimeError("存在未通过的边缘证据核验，真实数据已保存")


if __name__ == "__main__":
    main()
