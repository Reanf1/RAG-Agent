"""本地 RAG 请求快照日志与 Top-1 重排分数分布，不依赖额外服务。"""

from datetime import datetime
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
