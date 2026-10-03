"""50轮构造历史与真实本机Qwen的窗口核验；不是50轮真实模型对话或论文质量评测。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
from urllib.request import Request
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import BaseMessage
from src.agent import react_loop
from src.agent.memory import MemoryManager, count_history_tokens, run_session
from src.generation.rag_pipeline import urlopen
from src.utils.config import load_config


def serial(value):
    """保留真实事件和本机用量，Message转换为可存储结构。"""
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
        raise FileExistsError("请使用新输出路径保留原始核验数据")
    config = load_config()
    report = {"started_at": datetime.now().astimezone().isoformat(), "config": config,
              "seeded_turns_are_synthetic": True, "events": [], "model_requests": [], "validation_complete": False}
    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with tempfile.TemporaryDirectory() as directory:
        memory = MemoryManager(Path(directory) / "memory.sqlite3")
        session = memory.create_session("alice")
        alice_other = memory.create_session("alice")
        bob = memory.create_session("bob")
        for index in range(49):
            marker = "OLD_BEGIN_OUTSIDE_WINDOW" if index == 0 else f"ROUND_{index:02d}"
            memory.append_turn("alice", session, f"第{index}轮 {marker}：" + "这是明确构造的窗口功能测试历史，不是论文事实。" * 12,
                               f"第{index}轮构造回答：" + "记录此轮样例，只用于验证完整问答对的移除与保留。" * 12)
        code = "EXP_" + uuid4().hex[:12].upper()
        memory.append_turn("alice", session, f"最近的实验代号是{code}。", f"已记录本会话最近实验代号{code}。")
        memory.append_turn("alice", alice_other, "ALICE_OTHER_PRIVATE", "其他会话自己的回答")
        memory.append_turn("bob", bob, "BOB_PRIVATE", "Bob自己的回答")
        history = [{"role": m.type, "content": m.content} for m in memory.get_messages("alice", session)]
        context = memory.get_context("alice", session)
        report.update(archive_turns_before=50, archive_tokens_before=count_history_tokens(history),
                      window_before=context["history_window"], expected_code=code)
        save()
        # raw接口只用于核对JSON文本分词数，不把历史Token当成完整chat请求用量。
        text = json.dumps(context["history"], ensure_ascii=False)
        payload = {"model": config["llm"]["model"], "prompt": text, "raw": True, "stream": False,
                   "options": {"num_predict": 1, "num_ctx": config["llm"]["num_ctx"]}}
        request = Request(config["llm"]["base_url"].rstrip("/") + "/api/generate",
                          data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=300) as response:
            raw = json.load(response)
        report["raw_counter_check"] = {"model": raw.get("model"), "done": raw.get("done"),
                                       "local_json_tokens": count_history_tokens(context["history"]),
                                       "ollama_prompt_eval_count": raw.get("prompt_eval_count")}
        save()
        real_urlopen = react_loop.urlopen
        def record_request(request, *args, **kwargs):
            report["model_requests"].append(json.loads(request.data))
            save()
            return real_urlopen(request, *args, **kwargs)
        with patch("src.agent.react_loop.urlopen", side_effect=record_request):
            for event in run_session("我最近的实验代号是什么？请只回答代号。", "alice", session, tools=[], memory=memory):
                report["events"].append(serial(event))
                save()
                print(event["type"], event.get("full_response", ""), flush=True)
        done = report["events"][-1]
        report["archive_turns_after"] = len(memory.get_messages("alice", session)) // 2
        report["other_session_message_counts"] = [len(memory.get_messages("alice", alice_other)), len(memory.get_messages("bob", bob))]
        checks = {"window_under_budget": context["history_window"]["tokens"] <= config["memory"]["max_history_tokens"],
                  "old_turns_removed": context["history_window"]["dropped_turns"] > 0,
                  "archive_kept": report["archive_turns_after"] == 51,
                  "other_sessions_unchanged": report["other_session_message_counts"] == [2, 2],
                  "real_followup_answered": done["task_complete"] and code in done["full_response"],
                  "raw_counter_matches": raw.get("done") is True and raw.get("prompt_eval_count") == context["history_window"]["tokens"]}
        for index, payload in enumerate(report["model_requests"]):
            body = json.dumps(payload, ensure_ascii=False)
            sent = json.loads(payload["messages"][1]["content"])["context"]["history"]
            checks[f"request_{index}_isolated"] = all(marker not in body for marker in
                                                     ("OLD_BEGIN_OUTSIDE_WINDOW", "ALICE_OTHER_PRIVATE", "BOB_PRIVATE"))
            checks[f"request_{index}_budget"] = count_history_tokens(sent) <= config["memory"]["max_history_tokens"]
        report["checks"] = checks
        report["validation_complete"] = all(checks.values())
        report["finished_at"] = datetime.now().astimezone().isoformat()
        save()
    if not report["validation_complete"]:
        raise RuntimeError("存在未通过的窗口核验，原始数据已保存")


if __name__ == "__main__":
    main()
