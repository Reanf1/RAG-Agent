"""本地 Qwen 请求预算：完整消息／工具结构计数，输出窗口单独预留。"""


def request_tokens(payload: dict) -> int:
    """序列化全部输入并保守预留聊天模板开销；不是服务实际消耗指标。"""
    if "model" in payload and not payload["model"].startswith("qwen2.5:"):
        raise ValueError("请求Token计数仅适配本地Qwen2.5词表，更换模型须先适配词表")
    from src.agent.memory import count_history_tokens
    inputs = {key: payload[key] for key in ("messages", "tools", "format") if key in payload}
    # JSON包含角色、工具参数和分隔符，再为每条消息保留ChatML边界及末尾模板余量。
    return count_history_tokens(inputs) + 16 * len(payload.get("messages", [])) + 64


def check_request_budget(payload: dict) -> int:
    """请求不得侵占num_predict输出余量；词表缺失或模型不适配直接报错。"""
    options = payload["options"]
    budget = options["num_ctx"] - options["num_predict"]
    tokens = request_tokens(payload)
    if tokens > budget:
        raise ValueError(f"完整请求约{tokens} Token，输入预算为{budget} Token（已预留输出）；请缩短问题或证据")
    return tokens
