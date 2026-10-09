"""核对原有会话行摘要；仅输出哈希与数量，不保存真实对话。"""

from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import sqlite3
import sys

baseline_path, database_path, output_path = map(Path, sys.argv[1:4])
before = json.loads(baseline_path.read_text(encoding="utf-8"))
visitor = "f70c291392504088983428cd02a76af8"


def digest(rows):
    return sha256(json.dumps(sorted(rows, key=repr), ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


with sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
    # 只读事务冻结本次核对；其他访客可能继续添加会话和消息。
    db.execute("BEGIN")
    sessions = [list(row) for row in db.execute("SELECT * FROM sessions WHERE user_id != ?", (visitor,))]
    count = before["session_counts"]["sessions"]
    surplus = len(sessions) - count
    # 本次只核对少量新增会话；差异过大时明确无法核验，不作无限枚举。
    sessions_match = False
    if 0 <= surplus <= 3:
        for added in combinations(range(len(sessions)), surplus):
            if digest([row for i, row in enumerate(sessions) if i not in added]) == before["session_rows_sha256"]["sessions"]:
                sessions_match = True
                break
    messages = [list(row) for row in db.execute("SELECT * FROM messages ORDER BY id LIMIT ?", (before["session_counts"]["messages"],))]
    summaries = [list(row) for row in db.execute("SELECT * FROM summaries")]
    rag_history = [list(row) for row in db.execute("SELECT * FROM rag_history")]
    checks = {
        "original_sessions_present_unchanged": sessions_match,
        "original_message_rows_unchanged": digest(messages) == before["session_rows_sha256"]["messages"],
        "original_summary_row_present_unchanged": any(digest([row]) == before["session_rows_sha256"]["summaries"] for row in summaries),
        "rag_history_unchanged": digest(rag_history) == before["session_rows_sha256"]["rag_history"],
    }
    result = {"scope": "原有行内容核验；会话新增不误报为全库未变，真实对话不导出", "checks": checks,
              "original_counts": before["session_counts"],
              "current_counts": {table: db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
                                 for table in ("sessions", "messages", "summaries", "rag_history")},
              "excluded_test_visitor": visitor, "status": "passed" if all(checks.values()) else "not_fully_verified"}
output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
