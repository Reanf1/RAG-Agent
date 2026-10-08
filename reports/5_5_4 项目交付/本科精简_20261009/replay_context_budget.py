"""用当前消息构造重放历史F002，比较8K/16K输入预算，不连接模型。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FROZEN = ROOT / "reports/5_5_2 系统性能评估/Windows正式性能评测_20261007"
sys.path.insert(0, str(ROOT))
from langchain_core.messages import AIMessage, ToolMessage
from src.agent import react_loop
from src.agent.tools import get_available_tools
from src.utils.config import generation_options, load_config
from src.utils.messages import messages_to_ollama
from src.utils.token_budget import request_tokens

row = next(r for r in json.loads((FROZEN / "agent.json").read_text(encoding="utf-8"))["rows"]
           if r["id"] == "F002" and r["profile"] == "no_rules")
calls = [json.loads(line) for line in (FROZEN / "agent.calls.jsonl").read_text(encoding="utf-8").splitlines()]
call = next(c for c in calls if c["id"] == "F002" and c["profile"] == "no_rules"
            and [t["function"]["name"] for t in c["request"].get("tools", [])] == ["paper_summary"])
state = json.loads(next(m["content"] for m in call["request"]["messages"] if m["role"] == "user"))
event = {k: v for k, v in [e for e in row["metrics"]["trace"] if e["type"] == "tool_result"][-1].items()
         if k not in {"type", "iteration"}}
context = state["context"]
context["observations"].append(event)
messages = [AIMessage(content="", tool_calls=[{"name": event["name"], "args": event["args"], "id": event["call_id"]}]),
            ToolMessage(content=json.dumps(event["result"], ensure_ascii=False),
                        tool_call_id=event["call_id"], name=event["name"])]
question = next(q["question"] for q in json.loads((ROOT / "reports/评测集.json").read_text(encoding="utf-8"))
                if q["id"] == "F002")
results = []


class Captured(Exception):
    """请求取得后停止，避免后续网络调用。"""


def capture(model_messages, **fields):
    config = load_config()["llm"]
    for size in (8192, 16384):
        payload = {"model": config["model"], "stream": False,
                   "options": {**generation_options(config), "num_ctx": size}, **fields,
                   "messages": messages_to_ollama(model_messages)}
        original = request_tokens(payload)
        react_loop._fit_agent_payload(payload)
        fitted = request_tokens(payload)
        budget = size - payload["options"]["num_predict"]
        results.append({"num_ctx": size, "original_estimated_tokens": original,
                        "fitted_estimated_tokens": fitted, "input_budget": budget,
                        "characters_retained": sum(len(m.get("content", "")) for m in payload["messages"]),
                        "fits": fitted <= budget})
    raise Captured


before_context, before_messages = deepcopy(context), deepcopy(messages)
with patch.object(react_loop, "_model_request", capture), patch.object(react_loop, "route_question", lambda *a, **k: None):
    try:
        list(react_loop._observe_events(question, get_available_tools(), context, messages, thought=state["thought"]))
    except Captured:
        pass
assert context == before_context and messages == before_messages
assert len(results) == 2
print(json.dumps({"source_sha256": hashlib.sha256((ROOT / "src/agent/react_loop.py").read_bytes()).hexdigest(),
                  "context_and_native_messages_unchanged": True, "results": results}, ensure_ascii=False, indent=2))
