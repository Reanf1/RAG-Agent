"""构造长对话与真实本机Qwen核验：阶段压缩、旧事实追问、纠正与会话隔离。"""

import argparse
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from src.agent import react_loop
from src.agent.memory import MemoryManager, count_history_tokens, count_memory_tokens, run_session
from reports.verify_history_window import serial
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError("请使用新路径保留原始实测")
    report = {"started_at": datetime.now().astimezone().isoformat(), "config": load_config(),
              "seeded_turns_are_synthetic": True, "requests": [], "events": [], "checks": {}, "validation_complete": False}

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    real_urlopen = react_loop.urlopen
    def record(request, *args, **kwargs):
        item = {"request": json.loads(request.data)}
        report["requests"].append(item)
        save()
        started = perf_counter()
        try:
            with real_urlopen(request, *args, **kwargs) as response:
                raw = response.read()
            item.update(response=json.loads(raw), elapsed_seconds=perf_counter() - started)
            save()
            return BytesIO(raw)  # 保存真实HTTP原文，不构造模型结果。
        except Exception as error:
            item.update(error=str(error), elapsed_seconds=perf_counter() - started)
            save()
            raise

    def followup(memory, session, stage):
        for event in run_session("我当前的实验代号是什么？请只回答代号。", "alice", session, tools=[], memory=memory):
            report["events"].append({"stage": stage, **serial(event)})
            save()
            print(stage, event["type"], event.get("full_response", ""), flush=True)
        return report["events"][-1]

    with tempfile.TemporaryDirectory() as directory, patch("src.agent.react_loop.urlopen", side_effect=record):
        memory = MemoryManager(Path(directory) / "memory.sqlite3")
        session = memory.create_session("alice")
        other = memory.create_session("alice")
        bob = memory.create_session("bob")
        memory.append_turn("alice", other, "ALICE_OTHER_PRIVATE", "其他会话记录")
        memory.append_turn("bob", bob, "BOB_PRIVATE", "另一用户记录")
        code = "EXP_" + uuid4().hex[:10].upper()
        corrected = "EXP_" + uuid4().hex[:10].upper()
        report.update(expected_initial_code=code, expected_corrected_code=corrected)
        # 仅在旧轮次出现实验代号；最近四轮不提供此值，追问必须使用摘要。
        memory.append_turn("alice", session, f"我的人工智能课程实验代号是{code}，请记住。", f"用户已设定实验代号{code}。")
        for index in range(1, 12):
            memory.append_turn("alice", session, f"第{index}轮：请比较Transformer和CNN的结构，保持中文说明。" + "这里只验证摘要功能，并非科研事实评测。" * 7,
                               f"第{index}轮：后续要补实验结果和paper.pdf第3页引用；该段是构造回答。" + "不能把构造材料当项目实测结论。" * 7)
        archive = [{"role": m.type, "content": m.content} for m in memory.get_messages("alice", session)]
        report["archive_tokens_before"] = count_history_tokens(archive)
        context = memory.get_context("alice", session)
        report["first_context"] = context
        report["checks"].update(first_compressed=context.get("memory_summary", {}).get("summarized_turns") == 8,
                                 old_code_in_summary=code in context.get("summary", ""),
                                 old_code_absent_in_recent=code not in json.dumps(context["history"], ensure_ascii=False),
                                 first_under_budget=count_memory_tokens(context["history"], context.get("summary", "")) <= 2000)
        save()
        first = followup(memory, session, "initial")
        report["checks"]["first_followup"] = first["task_complete"] and first["full_response"].strip() == code
        # 第一轮追问后为5轮未压缩历史，再追加5轮达到第二次阶段阈值。
        memory.append_turn("alice", session, f"纠正：当前实验代号改为{corrected}，旧代号作废。", f"已记录用户纠正，当前实验代号为{corrected}。")
        for index in range(4):
            memory.append_turn("alice", session, f"新增第{index}轮：继续保留中文回答和真实引用要求。", "本轮没有新增实验代号。")
        updated = memory.get_context("alice", session)
        report["second_context"] = updated
        report["checks"].update(rolling_summary=updated.get("memory_summary", {}).get("summarized_turns") == 14,
                                 corrected_code_in_summary=corrected in updated.get("summary", ""),
                                 correction_absent_in_recent=corrected not in json.dumps(updated["history"], ensure_ascii=False),
                                 second_under_budget=count_memory_tokens(updated["history"], updated.get("summary", "")) <= 2000)
        before = len(report["requests"])
        reopened = MemoryManager(memory.db_path).get_context("alice", session)
        report["checks"]["restart_reuses_summary"] = reopened.get("summary") == updated.get("summary") and len(report["requests"]) == before
        second = followup(memory, session, "corrected")
        report["checks"]["corrected_followup"] = second["task_complete"] and second["full_response"].strip() == corrected
        report["archive_turns_after"] = len(memory.get_messages("alice", session)) // 2
        report["checks"]["archive_retained"] = report["archive_turns_after"] == 19
        report["checks"]["requests_isolated"] = all("PRIVATE" not in json.dumps(r["request"]) for r in report["requests"])
        memory.clear_session("alice", session)
        cleared = memory.get_context("alice", session)
        report["checks"]["clear_removes_summary_and_history"] = "summary" not in cleared and cleared["history"] == []
        report["checks"]["others_unchanged"] = [len(memory.get_messages("alice", other)), len(memory.get_messages("bob", bob))] == [2, 2]
        report["validation_complete"] = all(report["checks"].values())
        report["finished_at"] = datetime.now().astimezone().isoformat()
        save()
    if not report["validation_complete"]:
        raise RuntimeError("实测存在失败项，原始记录已保留")


if __name__ == "__main__":
    main()
