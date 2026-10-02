"""共享消息边界：内部使用LangChain Message，Context保存可序列化的历史字典。"""

from copy import deepcopy
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def messages_to_ollama(messages: list) -> list[dict]:
    """RAG、Agent和摘要共用同一原生消息转换；不把工具ID当作论文引用编号。"""
    result = []
    for message in messages:
        if not isinstance(message, (SystemMessage, HumanMessage, AIMessage, ToolMessage)) or not isinstance(message.content, str):
            raise ValueError("模型消息须为System/Human/AI/Tool Message，正文须为字符串")
        item = {"role": {"human": "user", "ai": "assistant"}.get(message.type, message.type),
                "content": message.content}
        if isinstance(message, AIMessage) and message.tool_calls:
            item["tool_calls"] = [{"function": {"name": call["name"], "arguments": deepcopy(call["args"])}}
                                  for call in message.tool_calls]
        if isinstance(message, ToolMessage):
            item["tool_name"] = message.name
        result.append(item)
    return result


def normalize_context(context: dict | None = None) -> dict:
    """统一历史为human/ai字典，保留摘要、检索引用和当前观察；返回独立快照。

    历史接收Human/AI Message或role/content字典；user/assistant在入口映射为human/ai。
    System及工具调用消息不进入长期历史，原生ToolMessage只用于当前轮Observation。
    RAG检索Context仍为普通字典，无历史时不添加无关字段或改变来源结构。
    """
    if context is not None and not isinstance(context, dict):
        raise ValueError("Context必须是字典")
    result = deepcopy(context) if context is not None else {}
    if "history" in result:
        if not isinstance(result["history"], list):
            raise ValueError("Context.history必须是消息列表")
        history = []
        for message in result["history"]:
            if isinstance(message, (HumanMessage, AIMessage)):
                if isinstance(message, AIMessage) and message.tool_calls:
                    raise ValueError("历史只保存最终问答，不保存中间工具调用")
                message = {"role": message.type, "content": message.content}
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError("历史消息须为Human/AI Message或仅含role/content的字典")
            if not isinstance(message["role"], str):
                raise ValueError("历史角色须为字符串")
            role = {"user": "human", "assistant": "ai"}.get(message["role"], message["role"])
            if role not in {"human", "ai"} or not isinstance(message["content"], str):
                raise ValueError("历史角色须为human/ai，正文须为字符串")
            history.append({"role": role, "content": message["content"]})
        result["history"] = history
    if "summary" in result and not isinstance(result["summary"], str):
        raise ValueError("Context.summary必须是字符串")
    # 原始Message仅留在事件中；Context不可含Message、Document、NaN等不能直接传输的值。
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Context须为可JSON序列化的字典，Message/Document请在入口转换") from error
    return result
