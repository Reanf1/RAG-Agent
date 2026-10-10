"""直接读取本地 Ollama NDJSON，以简单事件同步答案文本和引用证据。"""

import json
import re
from urllib.error import URLError

from src.generation.rag_pipeline import (
    _build_generation_request, _finish_generation, _with_generation_notice,
    generation_error, resolve_citations, urlopen,
)


def _visible_prefix(raw: str) -> str:
    """仅暂存末尾未闭合的引用编号或引用链接，其余Markdown直接展示。"""
    pending = re.search(r"(?<!\\)\[(?:参(?:考(?:文(?:档[0-9]*)?)?)?)?$|\[参考文档[0-9]+\][ \t]*(\([^\n)]*)$", raw)
    if pending:
        return raw[:pending.start(1) if pending.group(1) else pending.start()]
    return raw


def render_partial_answer(raw: str, context: dict) -> dict:
    """从完整原始前缀重算展示快照，不把已补全的编号再次当模型输入。"""
    visible = _visible_prefix(raw)
    if not visible.strip():
        return _with_generation_notice({"answer": "", "citations": [], "invalid_citation_ids": [],
                                        "missing_citations": False, "warnings": []}, context)
    resolved = resolve_citations(visible, context)
    # 生成时只显示正文；来源清单在 done 时统一补齐，证据列表可即时更新。
    resolved["answer"] = resolved["answer"].rsplit("\n\n## 参考来源\n", 1)[0]
    return _with_generation_notice(resolved, context)


def stream_answer(question: str, context: dict, *, options: dict | None = None):
    """依次产生 token/done/error 字典；只有收到服务 done 才返回最终结果。

    token 携带原始增量和已映射引用的正文快照；错误保留已收到的部分文本。
    HTTPResponse 按行解码 NDJSON，跨网络包的 UTF-8/JSON 由 readline 处理。
    关闭生成器或发生异常时 with 块关闭连接；不重试、补假答案或转云端。
    """
    raw, partial = "", render_partial_answer("", context)
    packet = None
    try:
        request, sampling = _build_generation_request(question, context, options, stream=True)
        with urlopen(request, timeout=300) as response:
            for line in response:
                if not line.strip():
                    continue
                packet = json.loads(line)
                if not isinstance(packet, dict) or not isinstance(packet.get("message", {}), dict):
                    raise ValueError("本地 Ollama 返回的数据格式无法解析")
                if packet.get("error"):
                    raise RuntimeError(f"本地 Ollama 返回错误：{packet['error']}")
                content = packet.get("message", {}).get("content", "")
                if not isinstance(content, str):
                    raise ValueError("本地 Ollama 返回的数据格式无法解析")
                if content:
                    raw += content
                    partial = render_partial_answer(raw, context)
                    yield {"type": "token", "content": content, **partial}
                if "done" in packet and type(packet["done"]) is not bool:
                    raise ValueError("本地 Ollama 的done字段必须为布尔值")
                if packet.get("done") is True:
                    # 末包通常不带正文；用累计文本，保留最后一包的真实统计。
                    visible = _visible_prefix(raw)
                    incomplete = visible != raw
                    result = _finish_generation({**packet, "message": {"content": visible if incomplete else raw}},
                                                context, sampling)
                    result["raw_answer"] = raw
                    if incomplete:
                        result["warnings"].append("回答末尾有未闭合的引用标记，已暂不展示该标记。")
                    yield {"type": "done", **result}
                    return
        raise RuntimeError("本地 Ollama 流已断开，未收到完成标记；回答尚未完成。")
    except (URLError, OSError, ValueError, RuntimeError) as error:
        event = {"type": "error", **generation_error(error), "raw_answer": raw, **partial}
        if isinstance(packet, dict) and any(key in packet for key in ("prompt_eval_count", "eval_count")):
            event["usage"] = {key: packet.get(key) for key in ("prompt_eval_count", "eval_count")}
        yield event
