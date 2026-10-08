"""上下文拼接、截断、本地生成共用逻辑与答案引用溯源。"""

from copy import deepcopy
from decimal import Decimal
import math
import re
import json
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from langchain_core.documents import Document
from markdown_it import MarkdownIt

from src.generation.prompt_template import NO_CONTEXT_TEXT, build_rag_messages
from src.retrieval.reranker import is_image_placeholder
from src.utils.config import generation_options, load_config, ollama_base_url
from src.utils.messages import messages_to_ollama
from src.utils.token_budget import check_request_budget, request_tokens

class _LocalModelRedirectHandler(HTTPRedirectHandler):
    """本地模型接口无需重定向；阻止目标绕过初始地址校验。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# 本机模型直接连接，并拒绝所有重定向，不向新的地址发送论文请求。
urlopen = build_opener(ProxyHandler({}), _LocalModelRedirectHandler()).open


def focus_answer_evidence(question: str, results: list[tuple[Document, float]], *, chunks=()) -> list[tuple[Document, float]]:
    """单一的使用／实现问题按连续双句精排，减少相邻对象混入；宽泛问题保留原块。"""
    if not re.search(r"(?:如何|怎样|怎么).*?(?:使用|实现|工作)|\bhow\b.*?\b(?:use|work|implement)\w*", question, re.I):
        return results
    if re.search(r"附录|消融|对比|比较|appendix|ablation|compar", question, re.I):
        return results
    from src.retrieval.bm25_retriever import expand_academic_query
    from src.retrieval.reranker import Reranker
    candidates, unchanged = [], []
    for document, score in results:
        text, metadata = document.page_content, document.metadata
        # 表格保留整行列关系；没有原文偏移的资料不能伪造摘录位置。
        if metadata.get("content_type") == "table" or type(metadata.get("start_index")) is not int:
            unchanged.append((document, score))
            continue
        boundaries = [0] + [match.end() for match in re.finditer(r"[.!?](?:\s+|$)|[。！？]\s*", text)]
        if boundaries[-1] < len(text):
            boundaries.append(len(text))
        if len(boundaries) <= 3:
            unchanged.append((document, score))
            continue
        for index in range(len(boundaries) - 2):
            start, end = boundaries[index], boundaries[index + 2]
            # 不改写原文；展示、引用和实际发给模型的区间完全一致。
            excerpt = deepcopy(document)
            excerpt.page_content = text[start:end]
            excerpt.metadata.update(start_index=metadata["start_index"] + start,
                                    end_index=metadata["start_index"] + end, evidence_excerpt=True)
            if "line_start" in metadata:
                excerpt.metadata.update(line_start=metadata["line_start"] + text[:start].count("\n"),
                                        line_end=metadata["line_start"] + text[:end].rstrip("\n").count("\n"))
            candidates.append((excerpt, 0.0))
    ranked = Reranker().rerank(expand_academic_query(question), candidates, k=len(candidates)) if candidates else []
    # 普通“如何使用”先选正文主设置；明确问消融／附录时保留相应范围。
    # 边界来自原文独立标题，不能把“see Appendix”当成正文结束。
    appendix = {}
    for doc in chunks:
        if re.search(r"(?m)^\s*(?:APPENDIX|附录)\s*$", doc.page_content):
            key, page = doc.metadata.get("doc_id"), doc.metadata.get("page_number")
            if type(page) is int:
                appendix[key] = min(appendix.get(key, page), page)
    ranked = sorted(ranked + unchanged, key=lambda pair: -pair[1])
    focused = {}
    threshold = load_config()["generation"]["low_relevance_threshold"]
    # “如何使用”优先明确的操作陈述；相似性图的观察不能当作模型执行步骤。
    operations = r"\b(?:we|our model)\s+(?:use|employ|adopt|add)\b|\bare added to\b|使用|采用|加入|添加"
    for document, score in ranked:
        key = document.metadata.get("doc_id")
        main = document.metadata.get("page_number", 1) < appendix.get(key, math.inf)
        if main and score >= threshold and re.search(operations, document.page_content, re.I):
            focused.setdefault(key, (document, score))
    for document, score in ranked:
        key = document.metadata.get("doc_id")
        main = document.metadata.get("page_number", 1) < appendix.get(key, math.inf)
        if main and score >= threshold:
            focused.setdefault(key, (document, score))
    for document, score in ranked:
        focused.setdefault(document.metadata.get("doc_id"), (document, score))
    return sorted(focused.values(), key=lambda pair: -pair[1])


def prepare_rag_context(question: str, results: list[tuple[Document, float]]) -> dict:
    """问答只接收 BGE sigmoid 重排结果；低相关性暂停生成，原文留待确认。"""
    context = build_context(question, results)
    threshold = load_config()["generation"]["low_relevance_threshold"]
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("low_relevance_threshold 必须在 (0, 1) 内")
    scores = [score for document, score in results if document.page_content.strip()]
    if any(not 0 <= score <= 1 for score in scores):
        raise ValueError("问答降级判断只接受 0~1 的模型重排分数，不能传入 BM25 或 RRF 分数")
    # 字符预算可能排除某些块，判断应依据实际送入模型的原文。
    top_score = max((reference["score"] for reference in context["references"]), default=None)
    mode = "empty" if top_score is None else "low" if top_score < threshold else "grounded"
    return {**context, "generation_mode": mode, "top_score": top_score, "threshold": threshold,
            "confirmed": False}


def _with_generation_notice(resolved: dict, context: dict) -> dict:
    """由系统保证提示出现，不依赖模型遵守提示词；历史与流式快照保持一致。"""
    mode = context.get("generation_mode", "grounded" if context.get("context", "").strip() else "empty")
    notice = ""
    if mode == "empty":
        notice = NO_CONTEXT_TEXT + "以下为纯模型回答，没有知识库文献依据。"
    elif mode == "low":
        notice = ("检索结果相关性低；已按你的确认使用候选内容，回答依据仍需核实。"
                  if context.get("confirmed") else "检索结果相关性低；尚未确认使用候选内容，未调用生成模型。")
    if notice and resolved["answer"]:
        resolved["answer"] = notice + "\n\n" + resolved["answer"]
    return {**resolved, "generation_mode": mode, "notice": notice}


def generation_error(error: Exception) -> dict:
    """把本地 API 故障转为可操作提示，原始详情单独保留，不自动重试。"""
    detail = f"{type(error).__name__}: {error}"
    if isinstance(error, HTTPError):
        # 404 也可能是接口路径错误，不能一律断言模型不存在。
        with error:
            detail += " " + error.read(4096).decode("utf-8", errors="replace")
        if error.code == 404:
            message, advice = "本地模型或接口未找到（HTTP 404）", "检查 config.yaml 中的模型名称和服务地址，用 ollama list 确认本地模型已准备好后重新提交问题。"
        elif error.code == 400:
            message, advice = "本地模型拒绝了请求（HTTP 400）", "检查模型名称、生成参数与请求格式，修正后重新提交问题。"
        else:
            message, advice = f"本地模型服务暂时无法完成请求（HTTP {error.code}）", "等待当前任务结束，检查本地服务日志与可用内存，必要时重启 Ollama 后重新提交问题。"
    elif isinstance(error, TimeoutError) or isinstance(error, URLError) and isinstance(error.reason, TimeoutError):
        message, advice = "本地模型响应超时", "等待模型加载完成，缩短问题或上下文，检查可用内存后重新提交问题。"
    elif isinstance(error, (URLError, OSError)):
        message, advice = "无法连接或读取本地 Ollama 服务", "启动本地 Ollama 服务并检查 config.yaml 中的地址和端口，恢复后重新提交问题。"
    elif isinstance(error, json.JSONDecodeError):
        message, advice = "本地模型返回的数据格式无法解析", "检查是否连接正确的 Ollama 接口及服务日志，恢复后重新提交问题。"
    elif isinstance(error, ValueError):
        message, advice = str(error), "检查本地配置和接口数据，修正后重新提交问题。"
    elif str(error).startswith("本地 Ollama 返回错误："):
        message, advice = "本地模型生成失败", "用 ollama list 检查模型已准备好，检查服务日志与可用内存，恢复后重新提交问题。"
    else:
        message, advice = str(error), "检查本地模型服务日志与可用内存，必要时重启 Ollama 后重新提交问题。"
    return {"message": message, "retry_advice": advice, "error_detail": detail}


def _build_generation_request(question: str, context: dict, options: dict | None = None,
                              *, stream: bool = False) -> tuple[Request, dict]:
    """流式和非流式共用消息及参数校验，避免出现两套模型配置。"""
    if context.get("generation_mode") == "low" and not context.get("confirmed"):
        raise ValueError("检索结果相关性低，请先查看候选原文并确认是否继续")
    config = load_config()["llm"]
    ollama_base_url(config)
    sampling = generation_options(config)
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
               "messages": messages_to_ollama(messages)}
    check_request_budget(payload)
    request = Request(config["base_url"].rstrip("/") + "/api/chat",
                      data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    return request, sampling


def generate_answer(question: str, context: dict, *, options: dict | None = None) -> dict:
    """按 YAML 调用本地非流式 Ollama，再用同轮 Context 补全文献引用。

    options 仅供参数实验覆盖；不包含检索 top_k，不创建额外业务服务。
    """
    request, sampling = _build_generation_request(question, context, options)
    result = None
    try:
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
        return _finish_generation(result, context, sampling)
    except (OSError, ValueError, RuntimeError) as error:
        failure = generation_error(error)
        exception = RuntimeError(f"本地 Ollama 调用失败：{failure['message']}。{failure['retry_advice']}")
        exception.usage = {key: result.get(key) for key in ("prompt_eval_count", "eval_count")} if isinstance(result, dict) else None
        raise exception from error


def _finish_generation(result: dict, context: dict, sampling: dict) -> dict:
    """两种生成方式共用终止检查、引用映射和服务实际用量。"""
    if not isinstance(result, dict) or not isinstance(result.get("message", {}), dict):
        raise ValueError("本地 Ollama 返回的数据格式无法解析")
    raw_answer = result.get("message", {}).get("content", "")
    if result.get("error"):
        raise RuntimeError(f"本地 Ollama 返回错误：{result['error']}")
    if not isinstance(raw_answer, str) or result.get("done") is not True or not raw_answer.strip():
        raise RuntimeError("本地 Ollama 未返回完整的非空答案")
    if not isinstance(result.get("model"), str) or not result["model"]:
        raise ValueError("本地 Ollama 返回的数据格式无法解析：缺少模型名称")
    resolved = resolve_citations(raw_answer, context)
    # 只核验模型明确声称逐字引用的句子；编号合法不代表解释语义正确。
    # 只归一PDF排版，不猜测或修补事实，失败时保留raw_answer供审核。
    references = {item["id"]: item["text"] for item in context["references"]}
    def quote_text(text):
        text = unicodedata.normalize("NFKC", text).casefold()
        # 仅去掉行末断词的连字符；同一行的术语连字符和数字原样保留。
        text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
        return re.sub(r"\s+", "", text)
    quote_errors, verified_quotes = [], []
    # 模型省略引号时也核验明确标为“原文依据”的内容，不能让数字改写绕过校验。
    for quote in re.finditer(r'(?:原文依据|Source evidence|Original evidence)\s*[:：]\s*([^\[]+?)\s*((?:\[参考文档\d+\])+)', raw_answer, re.I | re.S):
        evidence = quote[1].strip()
        if evidence.startswith(('"', '“')):
            evidence = evidence.rstrip('。.').strip()[1:]
            evidence = evidence[:-1] if evidence.endswith(('"', '”')) else evidence
        text = quote_text(evidence)
        ids = [int(value) for value in re.findall(r"\[参考文档(\d+)\]", quote[2])]
        if not text or not any(text in quote_text(references.get(value, "")) for value in ids):
            quote_errors.append(ids)
        else:
            verified_quotes.append((quote.group(), ids))
    if quote_errors:
        resolved["warnings"].append("回答声称引用的原句无法在所引片段定位，已暂停展示结论；请核对原文后重试。")
        resolved["answer"] = "当前回答的原文引句未通过定位校验，无法据此确认科研结论。请查看引用原文或补充相关片段。"
    resolved["evidence_quote_errors"] = quote_errors
    # 只核对带引用的“万／亿”数量换算；不能据此认定对象、条件或全部语义正确。
    # Decimal 按十进制比较，避免 303M 与 3.03 亿出现浮点误差。
    units = {"M": 1000000, "million": 1000000, "B": 1000000000,
             "billion": 1000000000, "万": 10000, "亿": 100000000}
    amount_pattern = r"(?<![\dA-Za-z.])(\d+(?:,\d{3})*(?:\.\d+)?)\s*(M\b|B\b|million\b|billion\b|万|亿)"
    def amounts(text):
        return {Decimal(number.replace(",", "")) * units[unit]
                for number, unit in re.findall(amount_pattern, text)}
    number_errors = []
    for claim in re.finditer(r"([^\n\[]*)((?:\[参考文档\d+\])+)", raw_answer):
        converted = amounts(" ".join(re.findall(r"[\d,]+(?:\.\d+)?\s*[万亿]", claim[1])))
        ids = [int(value) for value in re.findall(r"\[参考文档(\d+)\]", claim[2])]
        source = "\n".join(references.get(value, "") for value in ids)
        if converted and re.search(r"\d\s*(?:M\b|B\b|million\b|billion\b)", source):
            if not converted <= amounts(source):
                number_errors.extend(value for value in ids if value not in number_errors)
    if number_errors:
        resolved["warnings"].append("回答中的万／亿数量与所引原文不一致，已暂停展示结论；请核对原文数值与单位。")
        resolved["answer"] = "当前回答的数量换算未通过原文核对，无法据此确认科研结论。请查看引用原文中的数值与单位。"
        # 有已定位的引句时仍展示原文供核对，不改写模型的错误数量或冒充任务完成。
        excerpts = [text for text, ids in verified_quotes if set(ids) & set(number_errors)]
        if excerpts:
            resolved["answer"] = resolve_citations(resolved["answer"] + "\n\n" + "\n\n".join(excerpts), context)["answer"]
    resolved["evidence_number_errors"] = number_errors
    resolved = _with_generation_notice(resolved, context)
    if result.get("done_reason") == "length":
        resolved["warnings"].append("回答已达到生成 Token 上限，内容可能尚未完整。")
    return {**resolved, "raw_answer": raw_answer,
            "model": result["model"], "options": sampling,
            "done_reason": result.get("done_reason"),
            "usage": {key: result.get(key) for key in
                      ("prompt_eval_count", "eval_count", "total_duration", "load_duration",
                       "prompt_eval_duration", "eval_duration")}}


def build_context(question: str, results: list[tuple[Document, float]], *, max_context_chars: int | None = None) -> dict:
    """先沿用字符／句界预算，再按词表核对最终RAG消息；引用始终对应入选前缀。"""
    config = load_config()["llm"]
    sampling = generation_options(config)
    def payload(context):
        return {"model": config["model"], "options": sampling, "messages": messages_to_ollama(build_rag_messages(question, context))}
    context = _build_context_chars(question, results, max_context_chars=max_context_chars)
    # 固定提示和问题本身超限时不能通过删光证据来伪装为空库。
    check_request_budget(payload(""))
    budget = sampling["num_ctx"] - sampling["num_predict"]
    if request_tokens(payload(context["context"])) > budget:
        low, high, best = 1, context["context_budget_chars"], None
        while low <= high:
            middle = (low + high) // 2
            try:
                candidate = _build_context_chars(question, results, max_context_chars=middle)
            except ValueError:
                low = middle + 1
                continue
            if request_tokens(payload(candidate["context"])) <= budget:
                best, low = candidate, middle + 1
            else:
                high = middle - 1
        if best is None:
            raise ValueError("Token预算不足以容纳来源与正文，请缩短问题")
        context = best
    context["prompt_tokens_estimate"] = check_request_budget(payload(context["context"]))
    return context


def _build_context_chars(question: str, results: list[tuple[Document, float]], *, max_context_chars: int | None = None) -> dict:
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
    # 论文对比为两篇原文各分配一半预算；普通RAG仍沿用原来的配置。
    if max_context_chars is not None:
        if type(max_context_chars) is not int or max_context_chars <= 0:
            raise ValueError("本次上下文字符预算必须为正整数")
        budget = min(budget, max_context_chars)

    # 空白正文不作为依据；负分仍可排序，不设置未经评测的相关性阈值。
    candidates = [(document, score) for document, score in results
                  if document.page_content.strip() and not is_image_placeholder(document.page_content)]
    if not all(math.isfinite(score) for _, score in candidates):
        raise ValueError("检索分数必须为有限数值")
    candidates.sort(key=lambda item: item[1], reverse=True)
    parts, sources, references = [], [], []
    context_chars = 0
    partial_count = 0
    separator = "\n\n---\n\n"
    truncation_marker = "\n[正文已截断]"

    # 贪心拼接：高相关块优先占预算，预算不足时保留自然句边界内的前缀。
    # 来源标题与截断提示同样占字符预算，references只收实际进入Prompt的证据；
    # 因此答案编号不会指向被舍弃的块，字符预算也不会被引用头部额外撑大。
    for document, score in candidates:
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
        text = metadata.get("rerank_excerpt", document.page_content)
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
            "score": score,
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
    warnings = list(dict.fromkeys(ref.get("metadata", {}).get("retrieval_warning")
                                 for ref in references.values() if ref.get("metadata", {}).get("retrieval_warning")))
    citations, invalid_ids, labels = [], [], {}

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
    # 使用CommonMark位置判断缩进代码，四空格的嵌套列表正文仍可含有效引用。
    code_lines = {index for token in MarkdownIt().parse(answer)
                  if token.type in {"code_block", "fence"} and token.map
                  for index in range(*token.map)}
    for index, line in enumerate(answer.splitlines(keepends=True)):
        if index in code_lines:
            lines.append(line)
            continue
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
