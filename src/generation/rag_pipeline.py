"""上下文拼接、截断、本地生成共用逻辑与答案引用溯源。"""

from copy import deepcopy
import math
import re
import json
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langchain_core.documents import Document

from src.generation.prompt_template import NO_CONTEXT_TEXT, build_rag_messages
from src.utils.config import load_config


def _build_generation_request(question: str, context: dict, options: dict | None = None,
                              *, stream: bool = False) -> tuple[Request, dict]:
    """流式和非流式共用消息及参数校验，避免出现两套模型配置。"""
    config = load_config()["llm"]
    url = urlparse(config["base_url"])
    if config["provider"] != "ollama" or url.scheme != "http" or url.hostname not in {
        "localhost", "127.0.0.1", "::1"
    } or url.username or url.password or url.query or url.fragment:
        raise ValueError("本项目生成接口只允许本机 Ollama HTTP 服务")
    sampling = {key: config[key] for key in
                ("temperature", "top_p", "top_k", "num_ctx", "num_predict", "repeat_penalty")}
    if options:
        if set(options) - (set(sampling) | {"seed"}):
            raise ValueError("不支持的生成实验参数")
        sampling.update(options)
    for key in ("temperature", "top_p", "repeat_penalty"):
        value = sampling[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} 必须为有限数值")
    if sampling["temperature"] < 0 or not 0 < sampling["top_p"] <= 1 or sampling["repeat_penalty"] <= 0:
        raise ValueError("temperature 必须非负，top_p 在 (0, 1] 内，repeat_penalty 必须为正")
    for key in ("top_k", "num_ctx", "num_predict"):
        if type(sampling[key]) is not int or sampling[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if "seed" in sampling and type(sampling["seed"]) is not int:
        raise ValueError("seed 必须为整数")
    messages = build_rag_messages(question, context["context"])
    payload = {"model": config["model"], "stream": stream, "options": sampling,
               "messages": [{"role": "user" if message.type == "human" else message.type,
                             "content": message.content} for message in messages]}
    request = Request(config["base_url"].rstrip("/") + "/api/chat",
                      data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    return request, sampling


def generate_answer(question: str, context: dict, *, options: dict | None = None) -> dict:
    """按 YAML 调用本地非流式 Ollama，再用同轮 Context 补全文献引用。

    options 仅供参数实验覆盖；不包含检索 top_k，不创建额外业务服务。
    """
    request, sampling = _build_generation_request(question, context, options)
    try:
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
    except (URLError, TimeoutError) as error:
        raise RuntimeError("本地 Ollama 调用失败，请检查服务、模型和超时情况") from error
    return _finish_generation(result, context, sampling)


def _finish_generation(result: dict, context: dict, sampling: dict) -> dict:
    """两种生成方式共用终止检查、引用映射和服务实际用量。"""
    raw_answer = result.get("message", {}).get("content", "")
    if result.get("error") or not result.get("done") or not raw_answer.strip():
        raise RuntimeError("本地 Ollama 未返回完整的非空答案")
    resolved = resolve_citations(raw_answer, context)
    if result.get("done_reason") == "length":
        resolved["warnings"].append("回答已达到生成 Token 上限，内容可能尚未完整。")
    return {**resolved, "raw_answer": raw_answer,
            "model": result["model"], "options": sampling,
            "done_reason": result.get("done_reason"),
            "usage": {key: result.get(key) for key in
                      ("prompt_eval_count", "eval_count", "total_duration", "load_duration",
                       "prompt_eval_duration", "eval_duration")}}


def build_context(question: str, results: list[tuple[Document, float]]) -> dict:
    """把同一检索方式的 Document/分数列表组织为参考项目风格的 Context。

    分数越高越相关，同分保持输入顺序；只读原 Document。不调用检索或模型。
    字符预算包含来源和分隔符，并随系统规范及问题长度动态缩小；不是 Token 计数。
    """
    config = load_config()["generation"]
    for key in ("max_context_chars", "max_prompt_chars"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    # 使用真实模板计算固定开销，避免 Prompt 更新后长度预算失效。
    empty_messages = build_rag_messages(question)
    empty_prompt_chars = sum(len(message.content) for message in empty_messages)
    if empty_prompt_chars > config["max_prompt_chars"]:
        raise ValueError("用户问题与系统规范超过 Prompt 字符预算，请缩短问题或调整预算")
    fixed_chars = empty_prompt_chars - len(NO_CONTEXT_TEXT)
    budget = min(config["max_context_chars"], config["max_prompt_chars"] - fixed_chars)

    # 空白正文不作为依据；负分仍可排序，不设置未经评测的相关性阈值。
    candidates = [(document, score) for document, score in results if document.page_content.strip()]
    if not all(math.isfinite(score) for _, score in candidates):
        raise ValueError("检索分数必须为有限数值")
    candidates.sort(key=lambda item: item[1], reverse=True)
    parts, sources, references = [], [], []
    context_chars = 0
    partial_count = 0
    separator = "\n\n---\n\n"
    truncation_marker = "\n[正文已截断]"

    for document, _ in candidates:
        metadata = document.metadata
        filename = metadata.get("source_file") or "来源信息未提供"
        # 只使用加载器的真实位置；Word/文本没有物理页码，不推造页码。
        if "page_number" in metadata:
            location = f"第{metadata['page_number']}页（物理页码）"
            if metadata.get("page_end", metadata["page_number"]) != metadata["page_number"]:
                location = f"第{metadata['page_number']}–{metadata['page_end']}页（物理页码）"
        elif "paragraph_index" in metadata:
            location = f"段落{metadata['paragraph_index']}"
        elif "table_index" in metadata:
            location = f"表格{metadata['table_index']}"
        elif "line_start" in metadata:
            location = f"行{metadata['line_start']}–{metadata['line_end']}"
        else:
            location = "位置未记录"
        header = f"[参考文档{len(parts) + 1} - 来源: {filename}；原始块位置: {location}]\n"
        join_chars = len(separator) if parts else 0
        body_budget = budget - context_chars - join_chars - len(header)
        text = document.page_content
        partial = len(text) > body_budget
        if partial:
            # 不能只放来源而没有正文，也不能截断来源标记。尝试后续较短来源的块。
            limit = body_budget - len(truncation_marker)
            if limit <= 0:
                continue
            prefix = text[:limit].rstrip()
            boundaries = list(re.finditer(r"\n|[。！？]|[.!?](?=\s)", prefix))
            # 只在后半段寻找自然边界，避免为很早的句号浪费大部分空间。
            if boundaries and boundaries[-1].end() >= limit / 2:
                prefix = prefix[:boundaries[-1].end()].rstrip()
            if not prefix.strip():
                continue
            text = prefix + truncation_marker
        parts.append(header + text)
        context_chars += join_chars + len(parts[-1])
        partial_count += partial
        # 编号与实际送入 Prompt 的块一一对应，快照不随调用方修改原文而变化。
        references.append({
            "id": len(parts), "source_file": metadata.get("source_file"), "location": location,
            "text": text.removesuffix(truncation_marker) if partial else text,
            "metadata": deepcopy(metadata), "truncated": partial,
        })
        if metadata.get("source_file") and filename not in sources:
            sources.append(filename)

    if candidates and not parts:
        # 有检索结果却放不下，不把预算错误伪装成“知识库没有相关文档”。
        raise ValueError("上下文字符预算不足以容纳来源标记与正文，请调整预算或缩短问题")
    context = separator.join(parts)
    dropped_count = len(candidates) - len(parts)
    return {
        "context": context,
        "sources": sources,
        "chunk_count": len(parts),
        "references": references,
        "context_budget_chars": budget,
        "context_chars": len(context),
        "prompt_chars": fixed_chars + len(context or NO_CONTEXT_TEXT),
        "truncated": bool(partial_count or dropped_count),
        "dropped_count": dropped_count,
    }


def resolve_citations(answer: str, context: dict) -> dict:
    """从本轮 Context 补全答案引用，返回展示文本、证据和校验提示。

    只识别正文中的 [参考文档N]，不信任模型手写的参考来源列表。
    映射可证明编号来自已入选的文档块，不能证明该块语义支持答案中的结论。
    """
    if not answer.strip():
        raise ValueError("答案不能为空")
    references = {reference["id"]: reference for reference in context["references"]}
    citations, invalid_ids, warnings, labels = [], [], [], {}

    def replace(match):
        """只用本轮元数据渲染来源；未知编号明确标为无效。"""
        reference_id = int(match.group(1))
        if reference_id not in references:
            if reference_id not in invalid_ids:
                invalid_ids.append(reference_id)
                warnings.append(f"参考文档{reference_id}不在本轮上下文中，无法溯源。")
            return f"[无效引用：参考文档{reference_id}]"
        if reference_id not in labels:
            reference = references[reference_id]
            citations.append(deepcopy(reference))
            filename = reference["source_file"] or "来源信息未提供"
            label = f"{filename}；{reference['location']}"
            # 文件名中的 Markdown 字符只作为文字展示，不生成伪链接或 HTML。
            labels[reference_id] = re.sub(r"([\\`*_{}\[\]()<>#!|])", r"\\\1", label)
            if not reference["source_file"] or reference["location"] == "位置未记录":
                warnings.append(f"参考文档{reference_id}缺少文件名或位置，无法完整溯源。")
        return f"[参考文档{reference_id}：{labels[reference_id]}]"

    # 处理完整答案。跳过常见 Markdown 代码，避免把示例中的编号算作文献依据。
    lines = []
    fence = None
    # 模型附在编号后的普通 Markdown 链接也不作为可信来源链接使用。
    citation_pattern = re.compile(r"(?<!\\)\[参考文档([0-9]+)\](?:[ \t]*\([^\n)]*\))?")
    for line in answer.splitlines(keepends=True):
        marker = re.match(r" {0,3}(`{3,}|~{3,})", line)
        if marker:
            run = marker.group(1)
            if fence is None:
                fence = (run[0], len(run))
            elif run[0] == fence[0] and len(run) >= fence[1]:
                fence = None
            lines.append(line)
            continue
        if fence:
            lines.append(line)
            continue
        # 该标题是 Prompt 约定的末尾来源区；由真实映射重建，不解析模型的页码。
        if re.fullmatch(r"##[ \t]+(?:参考来源|References|Reference Sources|Sources)[ \t]*(?:\r?\n)?",
                        line, re.I):
            break
        cursor = 0
        for code in re.finditer(r"(`+).*?\1", line):
            lines.append(citation_pattern.sub(replace, line[cursor:code.start()]))
            lines.append(code.group())
            cursor = code.end()
        lines.append(citation_pattern.sub(replace, line[cursor:]))

    missing_citations = bool(references) and not citations
    if missing_citations:
        warnings.append("回答未提供有效文献引用，不能作为已溯源答案。")
    source_lines = [f"- [参考文档{ref['id']}] {labels[ref['id']]}"
                    + ("；正文已截断，仅展示已送入上下文的证据" if ref["truncated"] else "")
                    for ref in citations]
    rendered = "".join(lines).rstrip() + "\n\n## 参考来源\n" + ("\n".join(source_lines) or "无可引用来源")
    return {"answer": rendered, "citations": citations, "invalid_citation_ids": invalid_ids,
            "missing_citations": missing_citations, "warnings": warnings}
