"""按调用方提供的工具注册列表执行一次；八个科研工具将在后续实现。"""

from copy import deepcopy
import json
from time import perf_counter
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool


def execute_tool(name: str, args: dict, tools: list[BaseTool], call_id: str | None = None) -> dict:
    """参考上游注册表查找与invoke，保留真实结果、错误、耗时和关联消息。

    多余字段提前拒绝，必填项/类型由LangChain参数Schema校验；失败不重试。
    原始参数深拷贝，避免工具修改已经记录的调用参数。
    """
    call_id = call_id or uuid4().hex
    started = perf_counter()
    result, error, status = None, "", "success"
    try:
        registry = {item.name: item for item in tools}
        if len(registry) != len(tools):
            raise ValueError("工具名称不能重复")
        if name not in registry:
            raise ValueError(f"未知工具：{name}")
        selected = registry[name]
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是字典")
        if set(args) - set(selected.args):
            raise ValueError("工具参数包含未声明的字段")
        json.dumps(args, allow_nan=False)  # 非有限数值等非法JSON不能进入工具。
        result = selected.invoke(deepcopy(args))
    except Exception as exception:
        status, error = "error", f"{type(exception).__name__}: {exception}"
    return {"type": "tool_result", "call_id": call_id, "name": name, "args": deepcopy(args),
            "status": status, "result": result, "error": error,
            "elapsed_seconds": perf_counter() - started,
            "message": ToolMessage(content=error if error else str(result), tool_call_id=call_id,
                                   name=name, status=status)}
