"""模块四记忆/消息联调：真实SQLite、Qwen及计算器；临时代号是功能样例。"""

import argparse
from datetime import datetime
import json
from importlib import import_module
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

serial = import_module("reports.5_3_4 多轮对话记忆管理.verify_session_isolation").serial
from src.agent import MemoryManager, run_session
from src.agent import react_loop
from src.agent.memory import count_memory_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError("请换用新输出路径，保留已有核验")
    report = {"started_at": datetime.now().astimezone().isoformat(), "rows": [],
              "scope": "真实本机模型、SQLite和工具；旧12轮为明确构造的摘要功能样例，非论文质量评测。"}
    real_urlopen = react_loop.urlopen

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def check_case(name, stream, check):
        row = {"name": name, "events": [], "model_requests": []}
        report["rows"].append(row)
        def record_request(request, *args, **kwargs):
            row["model_requests"].append(json.loads(request.data))
            save()
            return real_urlopen(request, *args, **kwargs)  # 记录实际请求，不替换模型响应。
        with patch("src.agent.react_loop.urlopen", side_effect=record_request):
            for event in stream:
                row["events"].append(serial(event))
                save()
                print(name, event["type"], event.get("name", ""), flush=True)
        row["passed"] = bool(check(row["events"][-1], row))
        save()

    with tempfile.TemporaryDirectory(prefix="agent-memory-context-") as directory:
        path = Path(directory) / "memory.sqlite3"
        memory = MemoryManager(path)
        alice = memory.create_session("alice")
        bob = memory.create_session("bob")
        long_session = memory.create_session("alice")
        code = "EXP_CTX_" + uuid4().hex[:10].upper()
        old_code = "EXP_OLD_" + uuid4().hex[:10].upper()
        report.update(session_ids={"alice": alice, "bob": bob, "long": long_session}, code=code, old_code=old_code)
        check_case("remember_real_turn", run_session(
            f"请记住：本会话实验代号是{code}，实验系数为3.14。简短确认即可。", "alice", alice, tools=[], memory=memory),
            lambda done, row: done["task_complete"] and code in done["full_response"])
        memory = MemoryManager(path)  # 重建对象，验证历史来自SQLite而非进程缓存。
        check_case("reopened_history_calculator", run_session("请用计算器将我刚才的实验系数乘以2。", "alice", alice, memory=memory),
            lambda done, row: done["task_complete"] and "6.28" in done["full_response"] and any(
                e["type"] == "tool_result" and e["name"] == "calculator" and e["status"] == "success"
                and e["result"]["result"] == "6.28" for e in row["events"]))
        messages = memory.get_messages("alice", alice)
        check_case("direct_langchain_history", react_loop.run_react("我的实验代号是什么？仅返回代号。", tools=[], context={"history": messages}),
            lambda done, row: done["task_complete"] and code in done["full_response"] and all(
                isinstance(m, dict) and m["role"] in {"human", "ai"} for m in done["context"]["history"]))
        check_case("bob_has_no_alice_history", run_session("我的实验代号是什么？", "bob", bob, tools=[], memory=memory),
            lambda done, row: not done["task_complete"] and code not in json.dumps(row, ensure_ascii=False))
        for index in range(12):
            question = f"第{index}轮为人工构造的摘要功能样例。" + (f"实验代号是{old_code}。" if index == 0 else "")
            memory.append_turn("alice", long_session, question, "保持中文回答；构造历史不作为已核实论文事实。")
        check_case("summary_and_recent_history", run_session("最初的实验代号是什么？仅返回代号。", "alice", long_session, tools=[], memory=memory),
            lambda done, row: done["task_complete"] and old_code in done["full_response"] and
                done["context"]["memory_summary"]["summarized_turns"] == 8 and
                done["context"]["history_window"]["retained_turns"] == 4 and
                count_memory_tokens(done["context"]["history"], done["context"].get("summary", "")) <= 2000)
        try:
            list(run_session("读取Alice历史", "bob", alice, tools=[], memory=memory))
        except PermissionError as error:
            report["foreign_access"] = {"passed": True, "error": str(error)}
        else:
            report["foreign_access"] = {"passed": False}
        report["archive_roles"] = [message.type for message in memory.get_messages("alice", alice)]
        report["final_history_only"] = report["archive_roles"] == ["human", "ai"] * 2
        report["long_archive_turns"] = len(memory.get_messages("alice", long_session)) // 2
    report["finished_at"] = datetime.now().astimezone().isoformat()
    report["passed"] = all(row["passed"] for row in report["rows"]) and report["foreign_access"]["passed"] and report["final_history_only"]
    save()
    if not report["passed"]:
        raise RuntimeError("存在未通过的记忆/消息核验，实际请求和结果已保留")


if __name__ == "__main__":
    main()
