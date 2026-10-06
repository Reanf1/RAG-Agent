"""明确问题的快速路由、独立工具批次执行和有界超时恢复。"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from copy import deepcopy
import math
import re
from threading import Condition, Event
from time import perf_counter

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from src.agent.tools import execute_tool
from src.utils.config import load_config


# 整个Streamlit进程共用额度，超时的函数真正退出后才释放。
_execution_condition = Condition()
_active_executions = 0


def _release_execution(_future):
    global _active_executions
    with _execution_condition:
        _active_executions -= 1
        _execution_condition.notify_all()


def parallel_limit() -> int:
    """同轮调用上限同时约束线程数，避免本地模型资源被无界占用。"""
    limit = load_config()["agent"]["max_parallel_calls"]
    if type(limit) is not int or limit < 1:
        raise ValueError("agent.max_parallel_calls必须为正整数")
    return limit


def route_question(question: str, tools: list[BaseTool], context: dict | None = None) -> dict | None:
    """首轮明确意图返回零模型调用的Thought；模糊、依赖或已有会话状态回退模型。

    只选择实际传入工具，不填参数。parallel_tools非空表示独立批次；
    同一工具用于两篇论文时列表仅含一个名字，Action负责生成两套真实输入。
    """
    started = perf_counter()
    if not question.strip():
        raise ValueError("问题不能为空")
    has_state = context and any(context.get(key) for key in ("observations", "last_observation", "context", "history", "messages", "summary"))
    if re.search(r"先.*(?:再|然后)|根据.*结果|用.*结果|不要|无需|不能|别调用|\b(?:then|after|don't|do not)\b", question, re.I):
        return None
    names = {item.name for item in tools}
    explicit = sorted([name for name in names if re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", question)], key=question.index)
    patterns = {
        "paper_list": r"(?:文献|论文|文档)列表|(?:列出|列举).{0,12}(?:论文|文献|文档)[？?。]?$|(?:上传|入库)了哪些(?:论文|文献|文档)[？?。]?$|有哪些(?:论文|文献|文档)[？?。]?$|\blist (?:uploaded )?(?:papers?|documents?)\b",
        "current_time": r"(?:当前|现在|系统|今天).{0,6}(?:时间|日期|几点)|\bcurrent (?:time|date)\b|\btime now\b",
        "keyword_extract": r"提取.{0,8}(?:关键词|关键字)|\bextract.{0,25}keywords?\b",
        "paper_summary": r"(?:生成|结构化).{0,8}摘要|(?:总结|概括).{0,15}(?:论文|文献)|\bsummari[sz]e.{0,25}(?:paper|document)",
        "paper_metadata": r"元信息|元数据|论文.{0,8}(?:标题|作者|年份|DOI)|\bpaper metadata\b",
        "paper_compare": r"(?:对比|比较).{0,20}(?:论文|文献)|两篇论文.{0,12}(?:区别|差异)|\bcompare.{0,25}papers?\b",
        "web_search": r"(?:联网|上网|网上|网络).{0,8}(?:搜索|查询|查找|检索)|\bsearch (?:the )?(?:web|internet)\b|\bsearch online\b",
    }
    # “本文最新实验”属于本地证据；“最新论文”属于需要外部核验的信息。
    local = bool(re.search(r"知识库|本地|已上传|已入库|上传的|这篇|本篇|本文|该论文|指定论文|\b[0-9a-f]{64}\b|\b(?:this|uploaded) (?:paper|document)s?\b", question, re.I))
    fresh = bool(re.search(r"最新|最近|近期|实时|今年|\b(?:latest|recent|current advances)\b", question, re.I))
    web = bool(re.search(patterns["web_search"], question, re.I))
    if has_state and not local and not explicit and not web and re.search(r"代号|会话|对话|历史|\b(?:conversation|history)\b", question, re.I):
        return None  # “最近的代号”是历史追问，不是需要联网核验的近期资讯。
    if local and (web or fresh) and re.search(r"同时|以及|另外|并(?:且|联网)|\b(?:and|also)\b", question, re.I):
        return None  # 本地证据与外部进展的组合任务交给ReAct，不跳过其中一种来源。
    # 显式名称不能遮住问题中的第二种意图；组合任务继续交给ReAct规划。
    hits = list(dict.fromkeys([*explicit, *[name for name, pattern in patterns.items() if re.search(pattern, question, re.I)]]))
    if fresh and not local and not explicit and not any(name in hits for name in ("current_time", "keyword_extract", "paper_list")):
        hits = ["web_search"]
    if hits == ["web_search"] and "web_search" not in names:
        # 真实模型曾把禁用联网的最新论文问题改走RAG，明确结束比继续猜测可靠。
        return {"type": "thought", "thought": "联网搜索当前不可用，无法核验最新外部信息。请在config.yaml中启用agent.online_search_enabled后重试。",
                "next_step": "answer", "tool_name": None, "parallel_tools": [], "unavailable_tool": "web_search",
                "route": "unavailable", "model": None, "usage": {"prompt_eval_count": 0, "eval_count": 0},
                "elapsed_seconds": perf_counter() - started}
    # 旧对话只用于补全追问，不能覆盖最新问题中明确的工具/文档目标。
    # 本轮已经执行过工具时仍由模型判断后续步骤，避免重复路由已完成任务。
    has_execution = context and any(context.get(key) for key in ("observations", "last_observation", "context"))
    current_target = bool(explicit or re.search(r"(?<![A-Za-z0-9])[0-9a-f]{64}(?![A-Za-z0-9])|\.(?:pdf|docx|txt|md)(?=$|[^A-Za-z0-9])", question, re.I))
    if has_state and (has_execution or not current_target):
        return None
    if len(hits) > 1:
        return None  # 多种意图交给模型判断是否独立，不能只做其中一项。
    selected, batch, reason = None, [], ""
    if hits:
        if hits[0] not in names:
            return None
        selected, reason = hits[0], "明确工具意图，跳过模型规划，交由Action生成参数。"
        if re.search(r"分别|同时|各自|\beach\b|\bboth\b", question, re.I) and (
                len(set(re.findall(r"\b[0-9a-f]{64}\b", question))) > 1 or re.search(r"两篇|两份|两个|\bboth\b", question, re.I)):
            if selected not in {"paper_metadata", "paper_summary", "knowledge_base_search", "keyword_extract"} or parallel_limit() < 2:
                return None
            batch, reason = [selected], "分别处理两份已给定论文，Action可提出同一工具的独立调用。"
    elif local or re.search(r"文献|论文|\bpapers?\b", question, re.I):
        if "knowledge_base_search" not in names:
            return None
        selected, reason = "knowledge_base_search", "论文资料问题优先查询实际知识库。"
    elif re.search(r"最新|最近|实时|\b(?:latest|recent)\b", question, re.I):
        return None
    elif re.fullmatch(r"(?:什么是.{1,40}|.{1,40}是什么|what is .{1,40})[？?。]?", question.strip(), re.I):
        reason = "独立的一般概念问题，直接进入Observation回答。"
    else:
        expression = question.replace("乘以", "*").replace("除以", "/").replace("加", "+").replace("减", "-")
        if "calculator" not in names or not re.fullmatch(r"(?:请计算|计算|calculate)?\s*[\d\s.()+*/×÷-]+[？?。]?", expression, re.I) or not re.search(r"[+*/×÷-]", expression):
            return None
        selected, reason = "calculator", "明确算式路由到实际注册的计算器。"
    return {"type": "thought", "thought": reason, "next_step": "tool" if selected else "answer", "tool_name": selected, "parallel_tools": batch,
            "route": "rule", "model": None, "usage": {"prompt_eval_count": 0, "eval_count": 0},
            "elapsed_seconds": perf_counter() - started}


def recovery_limits() -> tuple[float, int, int]:
    """集中校验三个必要上限；错误配置不能静默变成无限等待或重试。"""
    config = load_config()["agent"]
    timeout, retries, repeats = (config[key] for key in
                                ("tool_timeout_seconds", "max_tool_retries", "max_repeated_calls"))
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("agent.tool_timeout_seconds必须为有限正数")
    if type(retries) is not int or retries < 0:
        raise ValueError("agent.max_tool_retries必须为非负整数")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("agent.max_repeated_calls必须为正整数")
    return timeout, retries, repeats


def _execute_attempts(call, tools, retries, deadline, stopped, attempts):
    """同一线程串行重试；已返回的TimeoutError才可重试，关闭/到期后不能再启动。"""
    started = perf_counter()
    for attempt in range(retries + 1):
        if attempt and (stopped.is_set() or perf_counter() >= deadline):
            break
        attempts.append({"attempt": attempt + 1, "status": "running"})
        result = execute_tool(call["name"], call["args"], tools, call["call_id"])
        attempts[-1] = {"attempt": attempt + 1, **{key: result[key] for key in
                        ("status", "error", "error_kind", "elapsed_seconds")}}
        if result["error_kind"] != "timeout":
            break
    return {**result, "attempts": deepcopy(attempts), "elapsed_seconds": perf_counter() - started,
            "_finished_at": perf_counter()}


def execute_calls(calls: list[dict], tools: list[BaseTool], *, parallel: bool = False):
    """有界等待、有限重试，按原调用顺序输出。超时后的迟到结果不会再送入Agent。

    同步线程无法被强杀；截止时间后停止等待及后续重试，run_react结束本次请求，
    保留其他已成功结果。关闭生成器不等待后台函数退出，也不再启动未执行的串行调用。
    """
    calls, tools = deepcopy(calls), list(tools)
    limit = parallel_limit()
    if not calls or len(calls) > limit:
        raise ValueError("独立批次调用数量超过上限或为空")
    if len({call["call_id"] for call in calls}) != len(calls):
        raise ValueError("同批工具call_id不能重复")
    mode = "parallel" if parallel and len(calls) > 1 else "serial"
    timeout, retries, _ = recovery_limits()
    pool, stopped = ThreadPoolExecutor(max_workers=len(calls) if mode == "parallel" else 1), Event()
    def submit(call):
        global _active_executions
        started, attempts = perf_counter(), []
        deadline = started + timeout
        with _execution_condition:
            available = _execution_condition.wait_for(lambda: _active_executions < limit,
                                                       timeout=max(0, deadline - perf_counter()))
            if not available:
                return call, None, started, deadline, attempts
            _active_executions += 1
        try:
            future = pool.submit(_execute_attempts, call, tools, retries, deadline, stopped, attempts)
        except BaseException:
            _release_execution(None)
            raise
        future.add_done_callback(_release_execution)
        return call, future, started, deadline, attempts
    try:
        pending = [submit(call) for call in calls] if mode == "parallel" else []
        for index, call in enumerate(calls):
            call, future, started, deadline, attempts = pending[index] if pending else submit(call)
            if future is None:
                error = "工具执行额度已占满（含尚未退出的超时函数），本次工具未启动；请稍后重试。"
                yield {"type": "tool_result", **call, "status": "error", "result": None, "error": error,
                       "error_kind": "capacity", "pending": False, "attempts": [], "execution_mode": mode,
                       "elapsed_seconds": perf_counter() - started,
                       "message": ToolMessage(content=error, tool_call_id=call["call_id"], name=call["name"], status="error")}
                if mode == "serial":
                    break
                continue
            try:
                result = future.result(timeout=max(0, deadline - perf_counter()))
                if result.pop("_finished_at") > deadline:
                    raise FutureTimeout  # 原顺序收取结果时，不能接受已超过自身期限的迟到成功。
            except FutureTimeout:
                stopped.set()
                future.cancel()  # 仅能取消尚未开始的任务，不能取消已经执行的函数。
                error = f"工具{call['name']}超过{timeout:g}秒等待上限，已跳过；后台函数可能仍在运行。"
                result = {"type": "tool_result", **call, "status": "error", "result": None,
                          "error": error, "error_kind": "deadline", "pending": True,
                          "attempts": deepcopy(attempts), "elapsed_seconds": perf_counter() - started,
                          "message": ToolMessage(content=error, tool_call_id=call["call_id"], name=call["name"], status="error")}
            yield {**result, "execution_mode": mode}
            if result.get("pending") and mode == "serial":
                break
    finally:
        stopped.set()
        pool.shutdown(wait=False, cancel_futures=True)
