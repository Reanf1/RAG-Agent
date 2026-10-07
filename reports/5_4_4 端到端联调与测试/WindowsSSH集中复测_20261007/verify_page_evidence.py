"""读取隔离页面会话，核验取消、确认和原文快照；不运行模型或修改数据库。"""

import argparse
import json
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--canceled", required=True)
    parser.add_argument("--confirmed", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("不覆盖已有页面证据")
    connection = sqlite3.connect(args.db.as_uri() + "?mode=ro", uri=True)
    try:
        rows = [{"session_id": session, "question": question, "answer": answer,
                 "event": json.loads(details)["event"]}
                for session, question, answer, details in connection.execute(
                    "SELECT a.session_id, h.content, a.content, a.details FROM messages a "
                    "JOIN messages h ON h.id=a.id-1 WHERE a.role='ai' ORDER BY a.id")]
    finally:
        connection.close()
    canceled = [row for row in rows if row["session_id"] == args.canceled]
    confirmed = [row for row in rows if row["session_id"] == args.confirmed]
    assert len(canceled) == 1 and len(confirmed) == 2
    pending, final = (row["event"] for row in confirmed)
    before, after = (event["context"]["observations"][-1] for event in (pending, final))
    checks = {
        "cancel_added_no_turn": len(canceled) == 1,
        "unconfirmed_zero_generation": all(value == 0 for value in before["result"]["usage"].values()),
        "pending_incomplete": not pending["task_complete"] and before["result"]["status"] == "needs_confirmation",
        "same_question_and_args": confirmed[0]["question"] == confirmed[1]["question"] and before["args"] == after["args"],
        "same_evidence_snapshot": before["result"]["papers"] == after["result"]["papers"],
        "confirmed_complete": final["task_complete"] and after["result"]["confirmed"] and after["result"]["status"] == "answered",
        "all_dimensions": not after["result"]["missing_dimensions"],
        "two_real_selections": len(after["result"]["model_calls"]) == 2,
    }
    report = {"checks": checks, "messages": rows, "passed": all(checks.values()),
              "boundary": "仅对应指定隔离会话；助手质量初评和用户终审另记。"}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "passed": report["passed"]}, ensure_ascii=False))
    assert report["passed"]


if __name__ == "__main__":
    main()
