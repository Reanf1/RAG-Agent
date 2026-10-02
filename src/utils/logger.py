"""本地 RAG 请求快照日志与 Top-1 重排分数分布，不依赖额外服务。"""

from datetime import datetime
from copy import deepcopy
import json
import math
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo

from src.utils.config import load_config


_LOG_LOCK = Lock()


def request_time() -> str:
    """统一用北京时间记录请求与日志日期。"""
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def _log_directory() -> Path:
    directory = Path(load_config()["paths"]["logs"])
    return directory if directory.is_absolute() else Path(__file__).resolve().parents[2] / directory


def record_rag_request(message: dict, status: str) -> dict:
    """追加完整请求快照；同一编号的开始、检索和结束记录可相互关联。

    不修改消息；失败时抛出异常，由页面明确提示。Token 缺失保持 None，
    只有缓存命中或尚未尝试生成时才确定为零，不用字符数猜测实际消耗。
    """
    timestamp = request_time()
    usage = message.get("usage") or {}
    skipped = message.get("cache", {}).get("hit") or not message.get("generation_attempted")
    input_tokens = 0 if skipped else usage.get("prompt_eval_count")
    output_tokens = 0 if skipped else usage.get("eval_count")
    total = input_tokens + output_tokens if type(input_tokens) is int and type(output_tokens) is int else None
    documents = message.get("retrieved_documents", [])
    info = message["request_info"]
    record = {
        "schema_version": 1, "request_id": message["request_id"], "session_id": message["session_id"],
        "timestamp": timestamp, "started_at": message["started_at"], "status": status,
        "question": message["question"],
        "retrieval": {"status": message.get("retrieval_status", "not_started"),
                      "method": "hybrid_rerank", "top_k": info["retrieval"]["top_k"],
                      "score_model": info["retrieval"]["reranker_model"],
                      "score_revision": info["retrieval"]["reranker_revision"],
                      "documents": documents, "top1_score": documents[0]["score"] if documents else None},
        "context": message.get("context"), "generation_mode": message.get("generation_mode"),
        "answer": message.get("answer", ""), "raw_answer": message.get("raw_answer", ""),
        "citations": message.get("citations", []), "warnings": message.get("warnings", []),
        "model": message.get("model", info["llm"]["model"]),
        "options": message.get("options"), "request_info": info,
        "done_reason": message.get("done_reason"), "cache": message.get("cache", {"hit": False}),
        "timing": {"retrieval_seconds": message.get("retrieval_seconds", 0.0),
                   "generation_seconds": message.get("generation_seconds", 0.0),
                   "response_seconds": message.get("elapsed_seconds", 0.0)},
        "tokens": {"input": input_tokens, "output": output_tokens, "total": total,
                   "source": "cache" if message.get("cache", {}).get("hit") else "not_called" if skipped
                   else "ollama" if total is not None else "unavailable"},
        "usage": usage, "original_usage": message.get("original_usage"),
        "error": message.get("error"), "error_detail": message.get("error_detail"),
        "retry_advice": message.get("retry_advice"),
    }
    line = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    directory = _log_directory()
    with _LOG_LOCK:
        directory.mkdir(parents=True, exist_ok=True)
        # 按日追加，不删除旧记录；单个 Streamlit 进程内的会话线程共享此锁。
        with (directory / f"rag_{timestamp[:10]}.jsonl").open("ab+") as output:
            output.seek(0, 2)
            if output.tell():
                output.seek(-1, 2)
                if output.read(1) != b"\n":
                    output.write(b"\n")  # 隔开意外断电留下的半行，保留它供诊断。
            output.write(line)
    return record


def read_rag_requests() -> tuple[list[dict], int]:
    """读取各请求的最新快照，返回损坏行数量；等待确认不会重复计数。"""
    latest, invalid = {}, 0
    with _LOG_LOCK:
        for path in sorted(_log_directory().glob("rag_*.jsonl")):
            with path.open("rb") as source:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict) or record.get("schema_version") != 1 \
                                or not isinstance(record.get("request_id"), str) \
                                or not isinstance(record.get("retrieval"), dict) \
                                or not all(isinstance(record["retrieval"].get(key), str)
                                           for key in ("status", "score_model", "score_revision")):
                            raise ValueError("请求日志结构无效")
                        latest[record["request_id"]] = record
                    except (ValueError, UnicodeDecodeError):
                        invalid += 1
    return list(latest.values()), invalid


def retrieval_request_metrics() -> dict:
    """候选命中率只描述非空返回，不能代替标注评测集的Hit@5。

    缓存、未执行和失败不进入命中率分母；同一请求的多份快照只计一次。
    """
    records, invalid = read_rag_requests()
    attempted = [r for r in records if not r.get("cache", {}).get("hit")
                 and r["retrieval"]["status"] in {"success", "empty", "error"}]
    completed = [r for r in attempted if r["retrieval"]["status"] != "error"]
    hits = sum(bool(r["retrieval"].get("documents")) for r in completed)
    def average(key):
        rows = [r for r in attempted if r.get("status") not in {"started", "retrieved"}] if key == "response_seconds" else attempted
        values = [r.get("timing", {}).get(key) for r in rows]
        values = [v for v in values if type(v) in (int, float) and math.isfinite(v) and v >= 0]
        return sum(values) / len(values) if values else None
    return {"attempts": len(attempted), "completed": len(completed), "hits": hits,
            "hit_rate": hits / len(completed) if completed else None,
            "failed": len(attempted) - len(completed), "invalid_lines": invalid,
            "retrieval_seconds": average("retrieval_seconds"), "response_seconds": average("response_seconds")}


def update_agent_metrics(metrics: dict, event: dict) -> dict:
    """按阶段快照汇总公开轨迹、工具执行指标和实际模型用量。

    工具返回的usage是内部模型用量，不与Action混淆。重试前的失败尝试
    未报告usage时，保留已知部分并把完整总量标为未知。
    """
    result = deepcopy(metrics)
    calls = result.setdefault("calls", [])
    kind, iteration = event["type"], event.get("iteration", 0)
    # 只记录公开决策说明与工具事实，不复制原生Message、整个Context或模型私有推理字段。
    if kind != "memory_summary":
        keys = ("type", "thought", "next_step", "tool_name", "parallel_tools", "route", "name", "call_id",
                "args", "result", "status", "error", "error_kind", "pending", "attempts", "execution_mode",
                "elapsed_seconds", "observation", "decision", "task_complete", "reason", "stop_reason",
                "full_response", "failed_tools", "available_alternatives", "retry_advice")
        trace = {key: deepcopy(event[key]) for key in keys if key in event}
        trace["iteration"] = event.get("iteration", event.get("iterations", 0))
        if isinstance(event.get("message"), str):
            trace["message"] = event["message"]  # 错误/恢复的公开说明；不保存AIMessage和ToolMessage。
        result.setdefault("trace", []).append(trace)
    tools = result.setdefault("tool_calls", [])
    if kind in {"tool_call", "tool_result"}:
        current = next((c for c in tools if c["call_id"] == event["call_id"]), None)
        if current is None:
            current = {"call_id": event["call_id"], "name": event["name"], "iteration": iteration,
                       "status": "pending", "seconds": None, "attempts": None, "execution_mode": None}
            tools.append(current)
        if kind == "tool_result":
            seconds = event.get("elapsed_seconds")
            current.update(status=event["status"],
                           seconds=seconds if type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0 else None,
                           attempts=len(event["attempts"]) if isinstance(event.get("attempts"), list) else None,
                           execution_mode=event.get("execution_mode"))
    # 每个逻辑调用只计一次；未返回不进成功率分母，重试不会伪造为多个独立调用。
    def summarize(rows):
        finished = [c for c in rows if c["status"] in {"success", "error"}]
        successes = sum(c["status"] == "success" for c in finished)
        durations = [c["seconds"] for c in finished if c["seconds"] is not None]
        return {"started": len(rows), "completed": len(finished), "successes": successes,
                "failures": len(finished) - successes, "pending": len(rows) - len(finished),
                "success_rate": successes / len(finished) if finished else None,
                "mean_seconds": sum(durations) / len(durations) if durations else None}
    result["tools"] = {**summarize(tools), "by_tool": [
        {"name": name, **summarize([c for c in tools if c["name"] == name])}
        for name in dict.fromkeys(c["name"] for c in tools)]}
    phase, tool, usage, identifier, uncertain = None, "—", event.get("usage"), "", False
    tool_call_ids = []
    if kind in {"thought", "observation", "memory_summary"}:
        phase = {"thought": "Thought", "observation": "Observation", "memory_summary": "记忆摘要"}[kind]
        identifier = f"{kind}:{iteration}"
    elif kind == "tool_call":
        identifier = f"action:{iteration}"
        if any(c["id"] == identifier for c in calls):
            return result  # 同一批次后续工具事件的usage为0，不再归属第二次模型调用。
        batch = getattr(event.get("message"), "tool_calls", [])
        tool = " + ".join(c["name"] for c in batch) or event["name"]
        tool_call_ids = [c["id"] for c in batch] or [event["call_id"]]
        phase = "Action（共享）" if len(batch) > 1 else "Action"
    elif kind == "tool_result":
        phase, tool, identifier = "工具内部", event["name"], event["call_id"]
        tool_call_ids = [identifier]
        payload = event.get("result")
        usage = payload.get("usage") if isinstance(payload, dict) else None
        uncertain = event.get("status") != "success" or len(event.get("attempts", [])) > 1
        if tool == "knowledge_base_search":
            retrieval = (payload or {}).get("retrieval", {}) if isinstance(payload, dict) else {}
            result.setdefault("retrievals", []).append({"call_id": identifier, **retrieval,
                "seconds": payload.get("retrieval_seconds") if isinstance(payload, dict) else None})
    elif kind == "error":
        phase, identifier, uncertain = "失败阶段（用量未完整报告）", f"error:{iteration}", True
    if phase:
        usage = usage if isinstance(usage, dict) else {}
        tokens = [usage.get(key) for key in ("prompt_eval_count", "eval_count")]
        tokens = [n if type(n) is int and n >= 0 else None for n in tokens]
        calls.append({"id": identifier, "iteration": iteration, "phase": phase, "tool": tool,
                      "tool_call_ids": tool_call_ids,
                      "input": tokens[0], "output": tokens[1], "incomplete": uncertain,
                      "seconds": event.get("elapsed_seconds")})
    known_input = sum(c["input"] or 0 for c in calls)
    known_output = sum(c["output"] or 0 for c in calls)
    unknown = sum(c["incomplete"] or c["input"] is None or c["output"] is None for c in calls)
    result["tokens"] = {"input": None if unknown else known_input, "output": None if unknown else known_output,
                        "total": None if unknown else known_input + known_output,
                        "known_total": known_input + known_output, "unknown_calls": unknown}
    return result


def record_agent_request(question: str, event: dict) -> None:
    """会话入口按阶段追加指标快照；页面重跑不再次执行或重复记账。"""
    timestamp = request_time()
    record = {"schema_version": 1, "request_id": event["request_id"], "timestamp": timestamp,
              "user_id": event["user_id"], "session_id": event["session_id"], "question": question,
              "event": event["type"], "iteration": event.get("iteration"), "metrics": event["metrics"],
              "stop_reason": event.get("stop_reason"), "task_complete": event.get("task_complete")}
    line = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    with _LOG_LOCK:
        directory = _log_directory()
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"agent_{timestamp[:10]}.jsonl").open("ab+") as output:
            output.seek(0, 2)
            if output.tell():
                output.seek(-1, 2)
                if output.read(1) != b"\n":
                    output.write(b"\n")
            output.write(line)


def retrieval_score_distribution() -> dict:
    """按模型/版本统计实际检索的原始 Top-1；空库、失败及缓存不补零分。"""
    records, invalid = read_rag_requests()
    groups = {}
    for record in records:
        retrieval = record["retrieval"]
        score = retrieval.get("top1_score")
        if retrieval.get("status") != "success" or type(score) not in (int, float) \
                or not math.isfinite(score) or not 0 <= score <= 1:
            continue
        key = (retrieval["score_model"], retrieval["score_revision"])
        groups.setdefault(key, []).append(score)
    distributions = []
    for (model, revision), scores in sorted(groups.items()):
        bins = [0] * 10
        for score in scores:
            bins[min(int(score * 10), 9)] += 1  # 1.0 包含在最后一个区间。
        distributions.append({"model": model, "revision": revision, "count": len(scores),
                              "mean": sum(scores) / len(scores), "min": min(scores), "max": max(scores),
                              "bins": [{"range": f"{index / 10:.1f}–{(index + 1) / 10:.1f}", "count": count}
                                       for index, count in enumerate(bins)]})
    return {"requests": len(records), "cache_hits": sum(bool(row.get("cache", {}).get("hit")) for row in records),
            "empty_retrievals": sum(row["retrieval"].get("status") == "empty" for row in records),
            "failed_retrievals": sum(row["retrieval"].get("status") == "error" for row in records),
            "invalid_lines": invalid, "distributions": distributions}
