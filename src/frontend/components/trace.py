"""将公开 Agent 事件转成轨迹关系图，不增加模型请求。"""

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
