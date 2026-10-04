"""整理公开 Agent 事件的展示与统计，不增加模型请求。"""

import json


def trace_graph(trace: list[dict]) -> str:
    """同轮工具从 Thought 分支，结果按调用ID连接并汇入 Observation。"""
    lines = ['digraph agent {', 'rankdir=LR;', 'node [shape=box, fontname="Arial"];']
    labels = {"thought": "Thought · 决策", "tool_call": "Action · 工具调用",
              "tool_result": "工具结果", "observation": "Observation · 判断",
              "done": "结束", "error": "错误", "recovery": "错误恢复", "action_skipped": "跳过调用"}
    previous = thought = None
    calls, results = {}, []
    for index, step in enumerate(trace):
        node = f"n{index}"
        kind = step["type"]
        label = labels.get(kind, kind)
        if step.get("name"):
            label += "\n" + str(step["name"])
        if kind == "observation":
            label += "\n" + str(step.get("decision", "未报告"))
        if kind == "done":
            label += "\n" + str(step.get("stop_reason", "未报告"))
        lines.append(f'{node} [label={json.dumps(label, ensure_ascii=False)}];')
        parents = [previous] if previous else []
        if kind == "thought":
            thought, calls, results = node, {}, []
        elif kind == "tool_call":
            parents = [thought] if thought else parents
            calls[step["call_id"]] = node
        elif kind == "tool_result":
            parents = [calls[step["call_id"]]] if step["call_id"] in calls else parents
            results.append(node)
        elif kind == "observation" and results:
            parents = results
        for parent in parents:
            lines.append(f'{parent} -> {node};')
        previous = node
    return '\n'.join(lines + ['}'])


def execution_rows(event: dict) -> list[dict]:
    """把已有阶段用量与工具结果合为七列，未知用量不冒充零。"""
    metrics = event.get("metrics", {})
    tools = {call["call_id"]: call for call in metrics.get("tool_calls", [])}
    # 最终未取得工具结果也标失败；原始pending记录仍保留，统计不假冒已返回。
    statuses = {"success": "成功", "error": "失败", "pending": "失败" if event.get("type") == "done" else "执行中"}
    rows = []
    shown = set()
    for call in metrics.get("calls", []):
        tool = tools.get(call["id"])
        status = "失败" if call["phase"].startswith("失败") else "成功"
        if tool:
            status = statuses[tool["status"]]
            shown.add(call["id"])
        seconds = call.get("seconds")
        rows.append({"阶段": call["phase"], "轮次": call["iteration"], "工具": call["tool"],
                     "输入Token": str(call["input"]) if call["input"] is not None else "未知",
                     "输出Token": str(call["output"]) if call["output"] is not None else "未知",
                     "耗时": f"{seconds:.3f} 秒" if seconds is not None else "未知", "状态": status})
    # 尚未返回的工具没有内部用量行；运行时展示事实上的执行中状态。
    for identifier, tool in tools.items():
        if identifier not in shown:
            seconds = tool.get("seconds")
            rows.append({"阶段": "工具执行", "轮次": tool["iteration"], "工具": tool["name"],
                         "输入Token": "未知", "输出Token": "未知",
                         "耗时": f"{seconds:.3f} 秒" if seconds is not None else "未知",
                         "状态": statuses[tool["status"]]})
    for step in metrics.get("trace", []):
        if step["type"] in {"action_skipped", "recovery"}:
            rows.append({"阶段": "Action（跳过）" if step["type"] == "action_skipped" else "错误恢复",
                         "轮次": step["iteration"], "工具": "—", "输入Token": "—", "输出Token": "—",
                         "耗时": "未知", "状态": "成功"})
    if event.get("type") == "done":
        rows.append({"阶段": "任务结束", "轮次": event.get("iterations", 0), "工具": "—",
                     "输入Token": "—", "输出Token": "—", "耗时": "—",
                     "状态": "成功" if event.get("task_complete") else "失败"})
    # 跳过/恢复没有用量行，按阶段位置插入，避免跳过Action排到Observation后。
    order = {"记忆摘要": 0, "Thought": 1, "Action": 2, "Action（共享）": 2, "Action（跳过）": 2,
             "工具内部": 3, "工具执行": 3, "Observation": 4, "错误恢复": 5, "任务结束": 7}
    return sorted(rows, key=lambda row: (row["轮次"], order.get(row["阶段"], 6)))


def conversation_statistics(messages: list[dict]) -> dict:
    """只累计当前会话每轮最终快照，不累加流式中间事件。"""
    metrics = [message.get("event", {}).get("metrics", {}) for message in messages]
    durations = [item["response_seconds"] for item in metrics if item.get("response_seconds") is not None]
    tokens = [item.get("tokens", {}) for item in metrics]
    known = sum(item.get("known_total", item.get("total") or 0) for item in tokens)
    unknown = sum(item.get("total") is None for item in tokens)
    calls = [call for item in metrics for call in item.get("tool_calls", [])]
    finished = [call for call in calls if call["status"] in {"success", "error"}]
    successes = sum(call["status"] == "success" for call in finished)
    tool_times = [call["seconds"] for call in finished if call.get("seconds") is not None]
    retrievals = [retrieval for item in metrics for retrieval in item.get("retrievals", [])]
    # 缓存和确认复用没有发起新检索，不进入检索次数或平均耗时。
    retrievals = [item for item in retrievals if item.get("status") in {"success", "empty", "error"}]
    retrieved = [item for item in retrievals if item.get("status") in {"success", "empty"}]
    hits = sum(item.get("returned_chunks", 0) > 0 for item in retrieved)
    retrieval_times = [item["seconds"] for item in retrievals if item.get("seconds") is not None]
    completions = [message for message in messages if isinstance(message.get("complete"), bool)]
    rounds = [message["event"]["iterations"] for message in messages if "iterations" in message.get("event", {})]
    return {"requests": len(messages), "successes": sum(message.get("complete") is True for message in messages),
            "completed_requests": len(completions), "known_tokens": known, "unknown_tokens": unknown, "tokens": None if unknown else known,
            "seconds": sum(durations) if len(durations) == len(messages) else None, "mean_seconds": sum(durations) / len(durations) if durations else None,
            "mean_rounds": sum(rounds) / len(rounds) if rounds else None,
            "tool_calls": len(calls), "tool_successes": successes, "tool_failures": len(finished) - successes,
            "tool_pending": len(calls) - len(finished), "tool_rate": successes / len(finished) if finished else None,
            "mean_tool_seconds": sum(tool_times) / len(tool_times) if tool_times else None,
            "retrieval_count": len(retrievals), "retrieval_hit_rate": hits / len(retrieved) if retrieved else None,
            "mean_retrieval_seconds": sum(retrieval_times) / len(retrieval_times) if retrieval_times else None}
