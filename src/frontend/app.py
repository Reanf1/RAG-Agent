"""批量导入、文档检索与本地 RAG 流式问答页面。"""

import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import streamlit as st


# 按文件位置导入项目模块，支持从其他目录启动 Streamlit。
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import LOADERS, create_import_tasks
from src.generation.cache import SemanticCache, cache_scope
from src.generation.prompt_template import PROMPT_VERSION
from src.generation.rag_pipeline import prepare_rag_context
from src.generation.streaming import stream_answer
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.logger import record_rag_request, request_time, retrieval_score_distribution
from src.retrieval.vector_store import VectorStore, batch_build_index
from src.utils.config import load_config

config = load_config()
app_config = config["app"]
raw_dir = project_root / config["paths"]["raw_documents"]
max_file_size_mb = config["importing"]["max_file_size_mb"]

st.set_page_config(page_title=app_config["name"], layout="wide")
st.title(app_config["name"])
st.caption(app_config["description"])
st.info("上传文档后增量写入本地知识库；下方 RAG 问答使用混合检索与模型重排，本地模型逐步输出答案并补全文献引用。")

# 导入状态仅存于当前页面会话，原始文件成功加载后保存到本地。
if "import_tasks" not in st.session_state:
    st.session_state.import_tasks = []
if "import_progress" not in st.session_state:
    st.session_state.import_progress = {"completed": 0, "total": 0}

with st.sidebar:
    st.header("文档导入")
    uploaded_files = st.file_uploader(
        "选择多份文档", type=[suffix.lstrip(".") for suffix in LOADERS],
        accept_multiple_files=True, max_upload_size=max_file_size_mb,
        key="document_uploads",
    )
    st.caption(f"支持 PDF、Word、TXT/Markdown，单份最大 {max_file_size_mb} MB。")
    selected_tasks = create_import_tasks([(file.name, file.getvalue()) for file in uploaded_files])
    same_selection = [t["id"] for t in selected_tasks] == [t["id"] for t in st.session_state.import_tasks]
    unfinished = any(t["status"] in {"pending", "loading", "chunking", "indexing"}
                     or (t["status"] == "success" and not t.get("indexed", False))
                     for t in st.session_state.import_tasks)
    start_import = st.button("开始导入", key="start_import",
                             disabled=not selected_tasks or (same_selection and not unfinished))
    retry_import = st.button(
        "重试失败项", key="retry_import",
        disabled=not any(t["status"] == "failed" for t in st.session_state.import_tasks),
    )

st.subheader("本批导入状态")
progress_bar = st.progress(0)
status_table = st.empty()
status_labels = {"pending": "等待", "loading": "加载中", "chunking": "分块中",
                 "indexing": "向量化与索引中", "success": "成功", "failed": "失败"}


def show_import_status():
    """刷新同一处进度和表格，不将内容单元误标为向量分块。"""
    progress = st.session_state.import_progress
    total = progress["total"]
    progress_bar.progress(progress["completed"] / total if total else 0,
                          text=f"已处理 {progress['completed']}/{total} 份")
    tasks = st.session_state.import_tasks
    if tasks:
        status_table.dataframe([{
            "文件": t["name"],
            "状态": "待索引" if t["status"] == "success" and not t.get("indexed", False)
                    else status_labels[t["status"]],
            "尝试次数": t["attempts"], "内容单元": len(t["documents"]),
            "分块数": t.get("chunk_count", 0), "已处理块": t.get("processed_chunks", 0),
            "本次新增块": t.get("added_chunks", 0),
            "写入后库中块数": str(t["index_total"]) if "index_total" in t else "—",
            "耗时（秒）": t["elapsed"], "错误": t["error"],
        } for t in tasks], hide_index=True, width="stretch")


if start_import and not same_selection:
    st.session_state.import_tasks = selected_tasks
if start_import or retry_import:
    for progress in batch_build_index(st.session_state.import_tasks, raw_dir,
                                     max_file_size_mb, retry_failed=retry_import):
        st.session_state.import_progress = progress
        show_import_status()
    # 完成后刷新按钮禁用状态，避免页面重跑或重复点击再次导入成功项。
    st.rerun()

show_import_status()
tasks = st.session_state.import_tasks
if tasks:
    successes = sum(t["status"] == "success" and t.get("indexed", False) for t in tasks)
    failures = sum(t["status"] == "failed" for t in tasks)
    st.caption(f"索引成功 {successes} 份，失败 {failures} 份。已处理块含跳过的重复项，新增块只计本次尝试；进度包含失败项。")
    if failures:
        st.warning("请根据错误信息修正文件或本地模型/数据库后重试；已保存原文和已写入块会保留。")
else:
    st.caption("在左侧选择文档后，点击“开始导入”。")

st.subheader("文档 Top-K 检索")
st.caption("此处只检索文档块；可选向量、BM25、RRF 或 RRF + 模型重排。各类分数不可直接比较，也不是命中概率。生成答案请使用下方 RAG 问答。")
with st.form("vector_search_form"):
    method = st.selectbox("检索方式", ["向量相似度", "BM25 关键词", "RRF 混合检索", "RRF + 模型重排"], key="retrieval_method")
    query = st.text_input("查询内容（支持中英文）", key="vector_query")
    top_k = st.number_input("返回数量 Top-K", min_value=1,
                            value=config["retrieval"]["top_k"], step=1, key="vector_top_k")
    doc_id = st.text_input("限定文档 ID（可选，留空检索全部文档）", key="vector_doc_id")
    search_submitted = st.form_submit_button("检索", key="vector_search")

if search_submitted:
    if not query.strip():
        st.warning("请输入查询内容。")
    else:
        try:
            # BM25 每次提交从当前正文重建小规模内存索引，无需加载 M3E。
            with st.spinner("正在检索本地知识库…"):
                if method in ("RRF 混合检索", "RRF + 模型重排"):
                    retriever = HybridRetriever()
                elif method == "BM25 关键词":
                    retriever = BM25Retriever()
                else:
                    retriever = VectorStore()
                if method == "RRF + 模型重排":
                    results = retriever.search(query, k=top_k, doc_id=doc_id.strip() or None, rerank=True)
                else:
                    results = retriever.search(query, k=top_k, doc_id=doc_id.strip() or None)
        except Exception as error:
            st.error(f"检索失败：{type(error).__name__}: {error}。请根据错误信息检查配置后重新检索。")
        else:
            if not results:
                st.info("没有可检索的文档块或关键词无匹配，请先导入文档并检查关键词；如填写了文档 ID，请检查是否正确。")
            st.caption(f"返回 {len(results)} 个文档块（Top-K={top_k}）。")
            for rank, (document, score) in enumerate(results, 1):
                metadata = document.metadata
                filename = metadata.get("source_file", "未知文件")
                score_label = {"向量相似度": "余弦相似度", "BM25 关键词": "BM25 分数",
                               "RRF 混合检索": "RRF 分数", "RRF + 模型重排": "模型相关性分数"}[method]
                with st.expander(f"{rank}. {filename} · {score_label} {score:.4f}", expanded=True):
                    # PDF 使用物理页码；Word/文本使用各自位置，不能伪造页码。
                    if "page_number" in metadata:
                        location = f"物理页码：{metadata['page_number']}"
                        if metadata.get("page_end", metadata["page_number"]) != metadata["page_number"]:
                            location += f"–{metadata['page_end']}"
                    elif "paragraph_index" in metadata:
                        location = f"段落：{metadata['paragraph_index']}"
                    elif "table_index" in metadata:
                        location = f"表格：{metadata['table_index']}"
                    elif "line_start" in metadata:
                        location = f"行范围：{metadata['line_start']}–{metadata['line_end']}"
                    else:
                        location = "位置未记录"
                    st.caption(f"来源：{filename}；{location}")
                    st.caption(f"文档 ID：{metadata.get('doc_id', '')}；块 ID：{metadata.get('chunk_id', '')}")
                    st.text(document.page_content)

def save_request(message, status):
    """日志故障明确提示，不能把已完成回答改成生成失败。"""
    try:
        record_rag_request(message, status)
        message.pop("log_error", None)
    except (OSError, ValueError, TypeError) as error:
        message["log_error"] = f"请求日志未保存：{type(error).__name__}: {error}"
        st.session_state.rag_log_error = message["log_error"]  # 清空或替换请求后也能看见故障。


st.subheader("RAG 流式问答")
st.caption("每个问题先匹配当前会话答案缓存；未命中再独立检索并由本地 Ollama 生成。历史仅保留展示，不作为多轮推理上下文。引用对应实际文件和位置，原文证据可展开查看。")
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []
if "rag_session_id" not in st.session_state:
    st.session_state.rag_session_id = uuid4().hex
if "rag_cache" not in st.session_state or st.session_state.rag_cache.settings != config["generation"]["cache"]:
    st.session_state.rag_cache = SemanticCache()
st.caption("清空当前对话会移除历史展示和答案缓存，本地请求日志会保留。")
if st.button("清空当前对话", key="clear_rag_chat"):
    if "rag_pending" in st.session_state:
        save_request(st.session_state.rag_pending["message"], "cancelled")
    st.session_state.rag_messages = []
    st.session_state.rag_cache.clear()
    st.session_state.pop("rag_pending", None)
    st.rerun()


def show_answer_details(message):
    """历史与本轮共用完成状态、实际用量和原文证据展示。"""
    if message.get("error"):
        st.error(f"回答未完成：{message['error']}。已保留部分文本。")
        st.info(message.get("retry_advice", "请检查本地服务或配置后重新提交问题。"))
    elif message.get("complete"):
        usage = message["usage"]
        status = "缓存已返回" if message.get("cache", {}).get("hit") else "服务已结束"
        st.caption(f"{status} · 检索 {message['retrieval_seconds']:.2f} 秒 · "
                   f"总耗时 {message['elapsed_seconds']:.2f} 秒 · "
                   f"输入 Token {usage['prompt_eval_count']} · 输出 Token {usage['eval_count']}")
        if message.get("cache", {}).get("hit"):
            hit = message["cache"]
            match = "相同问题" if hit["mode"] == "exact" else f"语义相似度 {hit['similarity']:.4f}"
            st.info(f"缓存命中（{match}），本次未调用检索、重排或生成模型。原问题：{hit['question']}")
    if message.get("notice"):
        st.info(message["notice"])
    if message.get("log_error"):
        st.warning(message["log_error"])
    for warning in message.get("warnings", []):
        st.warning(warning)
    for reference in message["citations"]:
        with st.expander(f"参考文档{reference['id']} · {reference['source_file'] or '来源信息未提供'} · {reference['location']}"):
            st.caption(f"块 ID：{reference['metadata'].get('chunk_id', '')}；仅展示本轮送入模型的原文证据。")
            st.text(reference["text"])


def generate_message(message, context, retriever, scope, placeholder):
    """正常回答和确认后的回答共用流式处理；未完成与低相关性答案不写缓存。"""
    placeholder.markdown("正在等待本地模型输出…")
    message["context"] = context
    message["generation_attempted"] = True
    started = perf_counter()
    try:
        for event in stream_answer(message["question"], context):
            if event["type"] == "token":
                message["answer"], message["citations"] = event["answer"], event["citations"]
                placeholder.markdown(event["answer"] + " ▌")
            else:
                message.update(event)
                message["complete"] = event["type"] == "done"
    finally:
        message["generation_seconds"] = perf_counter() - started
    if not message["complete"] and not message.get("message"):
        raise RuntimeError("生成流未返回完成标记")
    if message.get("type") == "error":
        message["error"] = message["message"]
    elif message["complete"] and context["generation_mode"] == "grounded":
        # 生成期间文献改变时不写入旧答案；缓存故障不影响已完成回答。
        try:
            if cache_scope(retriever.vector_store) == scope:
                st.session_state.rag_cache.put(message["question"], event, scope)
        except Exception as error:
            message["warnings"].append(f"本次答案已完成，但缓存未写入：{type(error).__name__}: {error}")


for message in st.session_state.rag_messages:
    with st.chat_message("user"):
        st.write(message["question"])
    with st.chat_message("assistant"):
        st.markdown(message["answer"])
        show_answer_details(message)

question = st.chat_input("询问已上传论文（支持中英文）", key="rag_question")
if question and question.strip():
    # 新问题取代旧的待确认请求，避免后来误点生成旧问题。
    if "rag_pending" in st.session_state:
        save_request(st.session_state.rag_pending["message"], "superseded")
    st.session_state.pop("rag_pending", None)
    message = {"question": question, "answer": "", "citations": [], "warnings": [], "complete": False,
               "request_id": uuid4().hex, "session_id": st.session_state.rag_session_id,
               "started_at": request_time(), "request_info": {"llm": deepcopy(config["llm"]),
               "retrieval": deepcopy(config["retrieval"]), "prompt_version": PROMPT_VERSION}}
    started = perf_counter()
    save_request(message, "started")
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        try:
            retriever = HybridRetriever()
            scope = cache_scope(retriever.vector_store)
            cached = st.session_state.rag_cache.lookup(question, scope)
            if cached:
                message.update(cached, complete=True, retrieval_seconds=0.0)
                message["retrieval_status"] = "skipped_cache"
            else:
                with st.spinner("正在检索本地文献…"):
                    search_started = perf_counter()
                    message["retrieval_status"] = "error"
                    try:
                        results = retriever.search(question, k=config["retrieval"]["top_k"], rerank=True)
                        message["retrieved_documents"] = [
                            {"rank": rank, "text": doc.page_content, "metadata": deepcopy(doc.metadata), "score": score}
                            for rank, (doc, score) in enumerate(results, 1)]
                        message["retrieval_status"] = "success" if results else "empty"
                        context = prepare_rag_context(question, results)
                    finally:
                        message["retrieval_seconds"] = perf_counter() - search_started
                message["context"], message["generation_mode"] = context, context["generation_mode"]
                message["elapsed_seconds"] = perf_counter() - started
                save_request(message, "awaiting_confirmation" if context["generation_mode"] == "low" else "retrieved")
                if context["generation_mode"] == "low":
                    st.session_state.rag_pending = {"message": message, "context": context, "scope": scope}
                else:
                    if context["generation_mode"] == "empty":
                        st.info("当前知识库中未找到相关文档；以下回答没有文献依据，将使用纯模型生成。")
                    generate_message(message, context, retriever, scope, answer_placeholder)
        except Exception as error:
            message["error"] = f"{type(error).__name__}: {error}"
        message["elapsed_seconds"] = perf_counter() - started
        answer_placeholder.markdown(message["answer"])
        if "rag_pending" not in st.session_state:
            save_request(message, "completed" if message["complete"] else "error")
            show_answer_details(message)
        elif message.get("log_error"):
            st.warning(message["log_error"])
    if "rag_pending" not in st.session_state:
        st.session_state.rag_messages.append(message)

if "rag_pending" in st.session_state:
    pending = st.session_state.rag_pending
    context = pending["context"]
    st.warning(f"检索结果相关性低：最高重排分数 {context['top_score']:.4f} < {context['threshold']}。"
               "请查看候选原文并确认是否使用；分数不是命中概率，确认也不代表原文能回答问题。")
    st.write(f"待确认问题：{pending['message']['question']}")
    for reference in context["references"]:
        with st.expander(f"候选{reference['id']} · {reference['source_file'] or '来源信息未提供'} · {reference['location']}", expanded=True):
            st.text(reference["text"])
    confirm = st.button("使用这些内容继续生成", key="confirm_low_relevance")
    cancel = st.button("取消本次回答", key="cancel_low_relevance")
    if confirm or cancel:
        st.session_state.pop("rag_pending")
        if cancel:
            save_request(pending["message"], "cancelled")
            if pending["message"].get("log_error"):
                st.warning(pending["message"]["log_error"])
            st.info("已取消本次回答。可以上传更相关的文献或重新描述问题。")
        else:
            message = pending["message"]
            started = perf_counter()
            with st.chat_message("assistant"):
                placeholder = st.empty()
                try:
                    retriever = HybridRetriever()
                    # 确认期间库或配置改变时，要求重提问题，不能使用过期原文。
                    if cache_scope(retriever.vector_store) != pending["scope"]:
                        raise ValueError("知识库或配置已改变，请重新提交问题并确认新的候选内容")
                    generate_message(message, {**context, "confirmed": True}, retriever, pending["scope"], placeholder)
                except Exception as error:
                    message["error"] = str(error)
                message["elapsed_seconds"] += perf_counter() - started  # 不计用户阅读等待时间。
                placeholder.markdown(message["answer"])
                save_request(message, "completed" if message["complete"] else "error")
                show_answer_details(message)
            st.session_state.rag_messages.append(message)

st.subheader("检索分数分布")
st.caption("统计本地请求日志中的实际 RAG 检索：Top-1 为 BGE sigmoid 重排分数，不是命中率或正确概率。空库、检索失败和缓存跳过均单独计数。")
try:
    statistics = retrieval_score_distribution()
    st.caption(f"请求 {statistics['requests']} · 缓存命中 {statistics['cache_hits']} · "
               f"空结果 {statistics['empty_retrievals']} · 检索失败 {statistics['failed_retrievals']}")
    if statistics["invalid_lines"]:
        st.warning(f"日志中有 {statistics['invalid_lines']} 行损坏或格式不符，统计已跳过并保留原文件。")
    for group in statistics["distributions"]:
        st.caption(f"{group['model']} · 版本 {group['revision'][:8]} · 样本 {group['count']} · "
                   f"均值 {group['mean']:.4f} · 范围 {group['min']:.4f}–{group['max']:.4f}")
        st.bar_chart(group["bins"], x="range", y="count", x_label="Top-1 分数区间", y_label="实际检索次数")
    if not statistics["distributions"]:
        st.info("尚无可统计的有结果 RAG 检索。提交问题并完成实际检索后显示分布。")
except (OSError, ValueError) as error:
    st.warning(f"无法读取检索统计：{type(error).__name__}: {error}。请检查本地日志目录。")
if "rag_log_error" in st.session_state:
    st.warning(st.session_state.pop("rag_log_error"))

st.subheader("模块开发状态")
st.table(
    [
        {"模块": "一：文档处理与检索", "状态": "部分实现", "范围": "已实现批量导入、分块、增量索引、向量/BM25/RRF 与模型重排；三档质量已评测，分块召回对比待完成"},
        {"模块": "二：RAG 生成", "状态": "已实现", "范围": "已实现 Prompt、上下文/引用、本地生成、流式、缓存、降级、请求日志与分数分布，完成参数对照；独立答案质量评测待完成"},
        {"模块": "三：Agent 决策", "状态": "未实现", "范围": "ReAct、工具、路由、恢复、记忆"},
        {"模块": "四：系统集成与前端", "状态": "部分实现", "范围": "已有文档入库、检索与 RAG 流式问答；Agent、持久会话、文献管理与完整联调待开发"},
        {"模块": "五：评测与交付", "状态": "部分实现", "范围": "已有三档检索实测、图表与 Excel；完整系统评测待完成"},
    ]
)
st.subheader("下一步")
st.write("后续继续分块召回对比和 Agent 等课程要求；独立答案质量评测仍保留。")
with st.sidebar:
    st.header("课程资料")
    st.write("南京农业大学生产实习课程实践")
    st.write("技术设计与验收：docs/技术设计文档.md")
