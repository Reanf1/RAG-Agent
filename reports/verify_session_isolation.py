"""真实SQLite与本地Qwen的三会话多轮核验，临时代号为功能样例，不是论文质量评测。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import BaseMessage
from src.agent import react_loop
from src.agent.memory import MemoryManager, run_session
from src.utils.config import load_config


def serial(value):
    """保存真实事件；原生Message转换为可写入JSON的结构。"""
    if isinstance(value, BaseMessage):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serial(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("请使用新输出路径，保留旧核验记录")
    report = {"started_at": datetime.now().astimezone().isoformat(), "config": load_config(),
              "rows": [], "validation_complete": False}
    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "memory.sqlite3"
        memory = MemoryManager(path)
        sessions = [{"user": user, "session": memory.create_session(user),
                     "code": "EXP_" + uuid4().hex[:12].upper()} for user in ("alice", "alice", "bob")]
        report["sessions"] = sessions
        real_urlopen = react_loop.urlopen
        def turn(identity, question, phase):
            row = {"name": phase, "user": identity["user"], "session": identity["session"],
                   "question": question, "events": [], "model_requests": []}
            report["rows"].append(row)
            def record_request(request, *args, **kwargs):
                payload = json.loads(request.data)
                row["model_requests"].append(payload)
                save()
                return real_urlopen(request, *args, **kwargs)
            with patch("src.agent.react_loop.urlopen", side_effect=record_request):
                for event in run_session(question, identity["user"], identity["session"], tools=[], memory=memory):
                    row["events"].append(serial(event))
                    save()
                    print(phase, identity["user"], event["type"], flush=True)
            done = row["events"][-1]
            row["passed"] = done["task_complete"] and done["stop_reason"] == "task_complete"
            for other in sessions:
                if other["session"] != identity["session"]:
                    row["passed"] &= other["code"] not in json.dumps(row, ensure_ascii=False)
            if phase == "followup":
                row["passed"] &= identity["code"] in done["full_response"]
            save()
        for identity in sessions:
            turn(identity, f"请记住本会话的实验代号是{identity['code']}，简短确认即可。", "remember")
        # 重新构造管理器，不依赖上一对象；历史从真实SQLite重新读取。
        memory = MemoryManager(path)
        for identity in reversed(sessions):
            turn(identity, "我刚才告诉你的实验代号是什么？请仅回答该代号。", "followup")
        try:
            list(run_session("读取Alice的历史", "bob", sessions[0]["session"], tools=[], memory=memory))
        except PermissionError as error:
            report["foreign_access"] = {"passed": True, "error": str(error)}
        else:
            report["foreign_access"] = {"passed": False}
        memory.clear_session("alice", sessions[0]["session"])
        report["message_counts_after_clear"] = [len(memory.get_messages(i["user"], i["session"])) for i in sessions]
        report["clear_isolated"] = report["message_counts_after_clear"] == [0, 4, 4]
        report["validation_complete"] = (all(row["passed"] for row in report["rows"])
                                           and report["foreign_access"]["passed"] and report["clear_isolated"])
        report["finished_at"] = datetime.now().astimezone().isoformat()
        save()
    if not report["validation_complete"]:
        raise RuntimeError("存在未通过的会话核验；原始请求、事件和回答已保留")


if __name__ == "__main__":
    main()
