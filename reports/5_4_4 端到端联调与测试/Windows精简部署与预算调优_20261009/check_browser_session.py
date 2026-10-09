"""只核对本轮独立网页会话的持久化指标，不导出其他用户历史。"""
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.project))
from src.utils.config import load_config

visitor = "bfc55e43e43546978ad7c56e34ba83a2"
session = "56ae86358a764fb3b112393489035d17"
# 使用只读连接，并先验证测试会话归属。
database = args.project / "data/sessions/memory.sqlite3"
with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
    assert connection.execute("SELECT user_id FROM sessions WHERE session_id=?", (session,)).fetchone() == (visitor,)
    answer, details = connection.execute(
        "SELECT content,details FROM messages WHERE session_id=? AND role='ai' ORDER BY id DESC LIMIT 1",
        (session,)).fetchone()
event = json.loads(details)["event"]
log_directory = Path(load_config()["paths"]["logs"])
if not log_directory.is_absolute():
    log_directory = args.project / log_directory
matching = []
for path in log_directory.glob("agent_*.jsonl"):
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("request_id") == event["request_id"] and record.get("event") == "done":
            matching.append(record)
assert len(matching) == 1
seconds = event["metrics"]["response_seconds"]
tokens = event["metrics"]["tokens"]["total"]
checks = {
    "database_log_seconds_equal": seconds == matching[0]["metrics"]["response_seconds"],
    "browser_seconds_match": f"{seconds:.3f}" == "16.291",
    "browser_tokens_match": tokens == 3214,
    "expected_answer": "WINMD20261005" in answer and "行1–6" in answer,
    "task_complete": event["task_complete"] is True,
}
result = {
    "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.project, text=True).strip(),
    "session_id": session, "request_id": event["request_id"],
    "response_seconds": seconds, "tokens": tokens,
    "checks": checks, "passed": all(checks.values()),
    "boundary": "本轮合成Markdown问答的网页、SQLite与日志核对；不是完整论文质量评审。",
}
with args.output.open("x", encoding="utf-8") as output:
    json.dump(result, output, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False))
raise SystemExit(0 if result["passed"] else 1)
