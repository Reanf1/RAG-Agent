"""本机真实边缘验收：空库、20MiB上传边界、长历史；逐步保存失败，不改业务数据。"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reports.verify_session_isolation import serial
from src.agent import react_loop
from src.agent.memory import MemoryManager, count_history_tokens, count_memory_tokens, run_session
from src.data_loader import batch_import, create_import_tasks
from src.generation import rag_pipeline
from src.generation.prompt_template import NO_CONTEXT_TEXT
from src.retrieval.vector_store import VectorStore, batch_build_index
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if root.exists() or output.exists():
        raise FileExistsError("请使用新的测试目录和报告路径，保留旧证据")
    root.mkdir(parents=True)
    config = deepcopy(load_config())
    baseline = {"config_sha256": sha256((ROOT / "config.yaml").read_bytes()).hexdigest(),
                "default_session_exists": (ROOT / config["paths"]["session_db"]).exists(),
                "index_ids": sorted(d.metadata["chunk_id"] for d in VectorStore().list_chunks())}
    for key, folder in (("raw_documents", "raw"), ("vector_index", "empty-index"), ("logs", "logs")):
        config["paths"][key] = str(root / folder)
    config["paths"]["session_db"] = str(root / "memory.sqlite3")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    report = {"started_at": datetime.now().astimezone().isoformat(), "config": deepcopy(config),
              "baseline": baseline, "rows": [], "checks": {}, "passed": False}
    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    active = {}
    real_http = react_loop.urlopen
    def record_http(request, *params, **kwargs):
        packet = {"request": json.loads(request.data)}
        active["model_packets"].append(packet)
        save()
        try:
            with real_http(request, *params, **kwargs) as response:
                raw = response.read()
            packet["response"] = json.loads(raw)
            save()
            return BytesIO(raw)  # 使用真实模型原始响应，仅额外记录请求/用量。
        except Exception as error:
            packet["error"] = f"{type(error).__name__}: {error}"
            save()
            raise
    with ExitStack() as stack:
        for module in ("src.utils.config", "src.agent.memory", "src.utils.logger", "src.agent.react_loop",
                       "src.agent.tools", "src.agent.router", "src.retrieval.vector_store",
                       "src.retrieval.hybrid_retriever", "src.retrieval.reranker", "src.generation.rag_pipeline",
                       "src.generation.cache", "src.chunking"):
            stack.enter_context(patch(module + ".load_config", return_value=config))
        for module in (react_loop, rag_pipeline):
            stack.enter_context(patch.object(module, "urlopen", side_effect=record_http))
        memory = MemoryManager()
        def turn(name, question, session=None, *, tools=None):
            nonlocal active
            session = session or memory.create_session("edge-verifier")
            active = {"name": name, "question": question, "session_id": session,
                      "events": [], "model_packets": [], "checks": {}}
            report["rows"].append(active)
            for event in run_session(question, "edge-verifier", session, tools, memory=memory):
                active["events"].append(serial(event))
                save()
                print(name, event["type"], event.get("stop_reason", ""), flush=True)
            active["history_after"] = serial(memory.get_messages("edge-verifier", session))
            return active

        row = turn("empty_knowledge_base", "知识库中的论文使用了什么实验数据集？请根据已上传原文给出名称和来源。")
        tool_results = [e["result"] for e in row["events"] if e["type"] == "tool_result" and e["name"] == "knowledge_base_search"]
        done = row["events"][-1]
        row["checks"] = {"real_empty_retrieval": bool(tool_results) and all(r["generation_mode"] == "empty" and
            r["retrieval"]["returned_chunks"] == 0 and not r["citations"] for r in tool_results),
            "explicit_tool_notice": all(NO_CONTEXT_TEXT in r["answer"] for r in tool_results),
            "explicit_final_notice": NO_CONTEXT_TEXT in done["full_response"],
            "no_invented_success": not done["task_complete"], "saved_turn": len(row["history_after"]) == 2,
            "no_web_search": not any(e.get("name") == "web_search" for e in row["events"]),
            "index_remains_empty": VectorStore().count() == 0}
        save()

        # PDF尾部补空白，不改页面正文；只用于文件字节大小边界，不冒充复杂巨型论文。
        import pymupdf
        pdf = pymupdf.open()
        pdf.new_page().insert_text((72, 72), "Upload size boundary fixture: local academic assistant.")
        raw_pdf = pdf.tobytes()
        pdf.close()
        limit = config["importing"]["max_file_size_mb"] * 1024 * 1024
        fixtures = root / "fixtures"
        fixtures.mkdir()
        for name, size in (("limit_minus_one.pdf", limit - 1), ("limit_exact.pdf", limit),
                           ("limit_plus_one.pdf", limit + 1), ("oversized_21MiB.pdf", 21 * 1024 * 1024)):
            data = raw_pdf + b" " * (size - len(raw_pdf))
            (fixtures / name).write_bytes(data)
            with pymupdf.open(stream=data, filetype="pdf") as document:
                assert document.page_count == 1 and "boundary fixture" in document[0].get_text()
        tasks = create_import_tasks([(name, (fixtures / name).read_bytes()) for name in
            ("limit_plus_one.pdf", "oversized_21MiB.pdf", "limit_minus_one.pdf", "limit_exact.pdf")])
        progress = list(batch_import(tasks, root / "upload-raw", config["importing"]["max_file_size_mb"]))
        store = VectorStore(root / "upload-index")
        index_progress = list(batch_build_index(tasks, root / "upload-raw", vector_store=store))
        list(batch_build_index(tasks, root / "upload-raw", retry_failed=True, vector_store=store))
        report["upload"] = {"sizes": [len(t["data"]) for t in tasks],
            "tasks": [{k: [d.model_dump() for d in v] if k == "documents" else serial(v)
                       for k, v in t.items() if k != "data"} for t in tasks],
            "progress": progress, "index_progress": index_progress, "index_count": store.count()}
        report["checks"].update(oversize_rejected=all(t["status"] == "failed" and "文件过大" in t["error"] and
                not t["path"] and not t["documents"] and not t.get("indexed") and t["attempts"] == 2 for t in tasks[:2]),
            size_boundary_accepted=all(t["status"] == "success" and t["indexed"] for t in tasks[2:]),
            exactly_two_valid_chunks=store.count() == 2,
            no_rejected_original=not any((root / "upload-raw").rglob("limit_plus_one.pdf")) and
                not any((root / "upload-raw").rglob("oversized_21MiB.pdf")))
        save()

        for name, turns in (("fifty_turn_history", 50), ("oversized_single_old_turn", 12)):
            session = memory.create_session("edge-verifier")
            code = "EDGE_" + uuid4().hex[:10].upper()
            for index in range(turns - 1):
                question = f"第{index}轮构造历史：保留中文回答与真实引用要求。" + "该段只用于长历史功能测试。" * 8
                answer = "已记录本轮测试需求；这不是实际论文结论。" * 4
                if name == "oversized_single_old_turn" and index == 0:
                    question += "超大旧问答内容。" * 3000
                memory.append_turn("edge-verifier", session, question, answer)
            memory.append_turn("edge-verifier", session, f"最近的实验代号改为{code}，请记住。", "已记录最新代号。")
            archive_before = serial(memory.get_messages("edge-verifier", session))
            row = turn(name, "我最近设置的实验代号是什么？请只回答代号。", session, tools=[])
            row.update(expected_code=code, seeded_turns=turns, seeded_history_is_synthetic=True,
                       archive_tokens_before=count_history_tokens([{"role": m["type"], "content": m["content"]} for m in archive_before]))
            # 从实际Thought请求取发送Context；不额外调用get_context触发下一批摘要。
            contexts = [json.loads(p["request"]["messages"][1]["content"])["context"] for p in row["model_packets"]
                        if "context" in json.loads(p["request"]["messages"][1]["content"])]
            row["contexts"] = contexts
            done = row["events"][-1]
            row["checks"] = {"real_followup": done["task_complete"] and done["full_response"].strip() == code,
                "window_bounded": bool(contexts) and all(count_memory_tokens(c["history"], c.get("summary", "")) <=
                    config["memory"]["max_history_tokens"] for c in contexts),
                "whole_turns_only": all(len(c["history"]) % 2 == 0 for c in contexts),
                "archive_preserved": row["history_after"][:-2] == archive_before and len(row["history_after"]) == 2 * (turns + 1),
                "history_truncated": all(c["history_window"]["dropped_turns"] > 0 for c in contexts),
                "no_log_error": not any(e.get("log_error") for e in row["events"])}
            if name == "fifty_turn_history":
                row["checks"]["summary_saved"] = bool(contexts[0].get("summary")) and any(
                    call.get("saved") for call in contexts[0].get("memory_summary", {}).get("calls", []))
            else:
                row["checks"]["large_old_turn_degraded"] = all("单轮超过" in c.get("memory_summary", {}).get("warning", "") and
                    "超大旧问答内容" not in json.dumps(c["history"], ensure_ascii=False) for c in contexts)
            save()
        report["agent_logs"] = [json.loads(line) for path in (root / "logs").glob("agent_*.jsonl")
                                for line in path.read_text().splitlines() if line.strip()]
        report["rag_logs"] = [json.loads(line) for path in (root / "logs").glob("rag_*.jsonl")
                              for line in path.read_text().splitlines() if line.strip()]
    report["checks"]["business_paths_unchanged"] = baseline == {
        "config_sha256": sha256((ROOT / "config.yaml").read_bytes()).hexdigest(),
        "default_session_exists": (ROOT / load_config()["paths"]["session_db"]).exists(),
        "index_ids": sorted(d.metadata["chunk_id"] for d in VectorStore().list_chunks())}
    report["finished_at"] = datetime.now().astimezone().isoformat()
    report["passed"] = all(report["checks"].values()) and all(all(r["checks"].values()) for r in report["rows"])
    save()
    if not report["passed"]:
        raise RuntimeError("边缘核验存在失败；真实事件、输入输出与保存记录均已保留")


if __name__ == "__main__":
    main()
