"""批量导入、文档检索与本地 RAG 流式问答页面。"""

import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import streamlit as st


# 按文件位置导入项目模块，支持从其他目录启动 Streamlit。
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import LOADERS, create_import_tasks
from src.frontend.components.documents import list_documents, delete_document, restore_document, read_pdf_page
from src.frontend.components.trace import trace_graph
from src.agent import run_session
from src.frontend.components.sessions import render_sessions
from src.generation.cache import SemanticCache, cache_scope
from src.generation.prompt_template import PROMPT_VERSION
from src.generation.rag_pipeline import prepare_rag_context
from src.generation.streaming import stream_answer
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.logger import (record_rag_request, request_time, retrieval_score_distribution,
                              retrieval_request_metrics)
from src.retrieval.vector_store import VectorStore, batch_build_index
from src.utils.config import check_health, load_config

config = load_config()
app_config = config["app"]


@st.dialog("引用原文页", width="large")
def show_original_page(reference):
    """直接打开引用对应的真实PDF页，并提供原文下载；不新增HTTP服务。"""
    try:
        page = read_pdf_page(project_root / config["paths"]["raw_documents"], reference)
        st.caption(f"{page['filename']} · 第{page['page_number']}页（物理页码）")
        st.image(page["image"], width="stretch")
        st.download_button("下载原始PDF", page["pdf"], file_name=page["filename"], mime="application/pdf")
    except (OSError, ValueError, KeyError) as error:
        st.error(f"无法打开引用原文：{error}")


def show_agent_sources(event):
    """本轮和持久化历史共用引用入口，各工具调用分别标识。"""
    for item in event.get("context", {}).get("observations", []):
        result = item.get("result")
        for reference in result.get("citations", []) if isinstance(result, dict) else []:
            if Path(reference["source_file"]).suffix.lower() == ".pdf" and "page_number" in reference["metadata"]:
                key = f"agent-citation:{event['request_id']}:{item.get('call_id')}:{reference['id']}"
                if st.button(f"查看{reference['source_file']} · {reference['location']}", key=key):
                    show_original_page(reference)


raw_dir = project_root / config["paths"]["raw_documents"]
index_dir = project_root / config["paths"]["vector_index"]
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
if "upload_version" not in st.session_state:
    st.session_state.upload_version = 0

with st.sidebar:
    st.header("文档上传与管理")
    uploaded_files = st.file_uploader(
        "选择多份文档", type=[suffix.lstrip(".") for suffix in LOADERS],
        accept_multiple_files=True, max_upload_size=max_file_size_mb,
        key=f"document_uploads:{st.session_state.upload_version}",
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

with st.sidebar:
    import_status = st.empty()
# 主区域知识库详情表仍需要完整状态名称，侧栏只呈现最终结果。
status_labels = {"pending": "等待", "loading": "加载中", "chunking": "分块中",
                 "indexing": "向量化与索引中", "success": "成功", "failed": "失败"}


def show_import_status():
    """开始导入后才显示进度，结果仅展示已经完成的成功和失败文件。"""
    progress = st.session_state.import_progress
    total = progress["total"]
    if not total:
        return
    tasks = st.session_state.import_tasks
    finished = [t for t in tasks if t["status"] == "failed" or
                (t["status"] == "success" and t.get("indexed", False))]
    with import_status.container():
        st.subheader("本批导入状态")
        st.progress(progress["completed"] / total,
                    text=f"已处理 {progress['completed']}/{total} 份")
        if finished:
            st.dataframe([{"文件": t["name"], "状态": "成功" if t["status"] == "success" else "失败"}
                          for t in finished], hide_index=True, width="stretch")
        successes = sum(t["status"] == "success" for t in finished)
        failures = len(finished) - successes
        st.caption(f"成功 {successes} 份，失败 {failures} 份。")
        if failures:
            with st.expander("失败详情"):
                for task in finished:
                    if task["status"] == "failed":
                        st.caption(f"{task['name']}：{task['error']}")


if start_import and not same_selection:
    st.session_state.import_tasks = selected_tasks
if start_import or retry_import:
    for progress in batch_build_index(st.session_state.import_tasks, raw_dir,
                                     max_file_size_mb, retry_failed=retry_import):
        st.session_state.import_progress = progress
        show_import_status()
    # 更换上传组件的key清空选择；失败任务保留，可继续重试。
    # 重试时若用户另选了新文件，不清掉尚未导入的新选择。
    if (start_import or same_selection) and any(
            t["status"] == "success" and t.get("indexed", False) for t in st.session_state.import_tasks):
        st.session_state.upload_version += 1
    st.rerun()

show_import_status()

library = None  # 读取失败保持未知，不能显示为正常空库。
library_error = ""
with st.sidebar:
    st.subheader("知识库文档")
    if "document_notice" in st.session_state:
        st.info(st.session_state.pop("document_notice"))
    try:
        try:
            library = list_documents(raw_dir, index_dir)
        except Exception:
            # 库读取失败时仍保留刷新入口，便于修复后重新读取。
            st.button("刷新知识库状态", key="refresh_knowledge")
            raise
        if not library:
            st.caption("知识库暂无文档。")
        for document in library:
            details, actions = st.columns([3, 1], vertical_alignment="center")
            with details:
                st.markdown(f"**{document['name']}**")
                st.caption(f"{document['doc_id'][:12]}… · {document['chunks']} 块 · "
                           + ("已向量化 · " if document["chunks"] else "未向量化 · ")
                           + ("原文已保存" if document["source_available"] else "原文缺失"))
            with actions:
                if st.button("删除", key=f"delete_document:{document['doc_id']}",
                             help=f"删除{document['name']}", disabled=not document["source_available"]):
                    st.session_state.delete_pending = document["doc_id"]
        # 刷新放在文档列表下方，确认删除与回收恢复操作上方。
        st.button("刷新知识库状态", key="refresh_knowledge")
        if library:
            documents_by_id = {document["doc_id"]: document for document in library}
            if "delete_pending" in st.session_state:
                pending_id = st.session_state.delete_pending
                st.warning(f"待删除：{documents_by_id.get(pending_id, {}).get('name', pending_id)}。确认后移除检索块，原文移入本地回收区。此操作影响共享知识库，已有对话保留。")
                confirm_delete = st.button("确认删除", key="confirm_delete_document")
                cancel_delete = st.button("取消删除", key="cancel_delete_document")
                if cancel_delete:
                    st.session_state.pop("delete_pending")
                    st.rerun()
                if confirm_delete:
                    removed = delete_document(raw_dir, index_dir, pending_id)
                    # 删除后不能复用“已索引成功”的上传任务；重新上传应重新入库。
                    st.session_state.import_tasks = [t for t in st.session_state.import_tasks if sha256(t["data"]).hexdigest() != pending_id]
                    st.session_state.import_progress = {
                        "completed": sum(t["status"] in {"success", "failed"} for t in st.session_state.import_tasks),
                        "total": len(st.session_state.import_tasks)}
                    if "rag_cache" in st.session_state:
                        st.session_state.rag_cache.clear()
                    st.session_state.pop("delete_pending")
                    st.session_state.document_notice = f"已删除 {removed} 个检索块，原文已回收，可在下方恢复。"
                    st.rerun()
        trash_dir = raw_dir / ".trash"
        archived = sorted(folder.name for folder in trash_dir.iterdir()
                          if folder.is_dir() and not folder.is_symlink() and len(folder.name) == 64
                          and all(c in "0123456789abcdef" for c in folder.name)) if trash_dir.is_dir() and not trash_dir.is_symlink() else []
        if archived:
            restore_id = st.selectbox("回收区文档ID", archived, key="restore_doc_id")
            if st.button("恢复文档并重建其索引", key="restore_document"):
                st.session_state.import_tasks = restore_document(raw_dir, restore_id)
                for progress in batch_build_index(st.session_state.import_tasks, raw_dir, max_file_size_mb):
                    st.session_state.import_progress = progress
                    show_import_status()
                if "rag_cache" in st.session_state:
                    st.session_state.rag_cache.clear()
                st.session_state.document_notice = "原文已恢复，请查看本批导入状态；失败项可重试。"
                st.rerun()
    except Exception as error:
        library_error = f"文档管理失败：{type(error).__name__}: {error}。原文保留，请修正后重试。"
        st.error(library_error)

def save_request(message, status):
    """日志故障明确提示，不能把已完成回答改成生成失败。"""
    try:
        record_rag_request(message, status)
        message.pop("log_error", None)
    except (OSError, ValueError, TypeError) as error:
        message["log_error"] = f"请求日志未保存：{type(error).__name__}: {error}"
        st.session_state.rag_log_error = message["log_error"]  # 清空或替换请求后也能看见故障。


session_ready = render_sessions(project_root / config["paths"]["session_db"], save_request)


def save_rag_message(message):
    """完成/错误回答持久化，保存失败明确提示；不重新生成答案。"""
    st.session_state.rag_messages.append(message)
    try:
        st.session_state.agent_memory.append_rag_message(st.session_state.agent_user_id,
                                                       st.session_state.rag_session_id, message)
    except Exception as error:
        st.warning(f"RAG历史未保存：{type(error).__name__}: {error}。本页答案仍保留，请检查会话数据库。")
    else:
        st.rerun()  # 更新侧栏会话标题，重绘已有答案，不重新生成。


st.subheader("科研对话")
agent_tab, rag_tab = st.tabs(["Agent 科研助理", "RAG 流式问答"])
with agent_tab:
    st.subheader("Agent 科研助理")
    st.caption("本页独立会话接入历史记忆；每个阶段完成即刷新实际 Token、工具成功率/耗时和决策轨迹。Action负责选择工具，内部用量单列；并行批次共享的Action只统计一次。左侧可新建、切换、删除或恢复历史会话。")
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    for previous in st.session_state.agent_messages:
        with st.chat_message("user"):
            st.markdown(previous["question"])
        with st.chat_message("assistant"):
            st.markdown(previous["answer"])
            if previous["complete"] is False:
                st.warning(f"任务未完成：{previous['stop_reason']}")
            show_agent_sources(previous.get("event", {}))
    agent_user_panel = st.empty()
    agent_answer_panel = st.empty()
    with st.form("agent_metrics_form"):
        agent_question = st.text_input("向 Agent 提问", key="agent_question")
        agent_submitted = st.form_submit_button("运行 Agent", key="run_agent", disabled=not session_ready)


with rag_tab:
    st.subheader("RAG 流式问答")
    st.caption("每个问题先匹配当前会话答案缓存；未命中再独立检索并由本地 Ollama 生成。历史和引用保存在当前会话，只用于展示，不作为RAG多轮推理上下文。引用对应实际文件和位置，原文证据可展开查看。")
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if "rag_cache" not in st.session_state or st.session_state.rag_cache.settings != config["generation"]["cache"]:
        st.session_state.rag_cache = SemanticCache()
    st.caption("清空会删除当前会话的RAG历史和答案缓存，Agent历史、知识库和请求日志保留。")
    if st.button("清空当前对话", key="clear_rag_chat", disabled=not session_ready):
        try:
            st.session_state.agent_memory.clear_rag_messages(st.session_state.agent_user_id, st.session_state.rag_session_id)
        except Exception as error:
            st.error(f"RAG历史清空失败：{type(error).__name__}: {error}。历史和缓存保留，可修复后重试。")
        else:
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
        if message.get("generation_mode") == "empty":
            st.info("当前知识库中未找到相关文档；以下回答没有文献依据，为纯模型回答。")
        elif message.get("notice"):
            st.info(message["notice"])
        if message.get("log_error"):
            st.warning(message["log_error"])
        for warning in message.get("warnings", []):
            st.warning(warning)
        for reference in message["citations"]:
            with st.expander(f"参考文档{reference['id']} · {reference['source_file'] or '来源信息未提供'} · {reference['location']}"):
                st.caption(f"块 ID：{reference['metadata'].get('chunk_id', '')}；仅展示本轮送入模型的原文证据。")
                st.text(reference["text"])
                if Path(reference["source_file"]).suffix.lower() == ".pdf" and "page_number" in reference["metadata"]:
                    key = f"citation:{message.get('request_id', id(message))}:{reference['id']}"
                    if st.button("查看引用原文页", key=key):
                        show_original_page(reference)


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

    question = st.chat_input("询问已上传论文（支持中英文）", key="rag_question", disabled=not session_ready)
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
            save_rag_message(message)

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
                save_rag_message(message)


st.subheader("Agent 决策轨迹与运行指标")
agent_metrics_panel = st.empty()
st.subheader("RAG 运行指标")
st.caption("候选命中率 = 非空检索次数 / 已完成检索次数；仅表示找到文档块，不代表回答正确或 Hit@5。缓存、未执行和检索失败不计入分母。统计包含直接 RAG 问答和 Agent 的知识库工具。")
rag_metrics_panel = st.empty()


def show_retrieval_metrics():
    """模型和工具事件完成后刷新日志统计，不启动额外检索或模型请求。"""
    with rag_metrics_panel.container():
        try:
            stats = retrieval_request_metrics()
            columns = st.columns(3)
            columns[0].metric("RAG 候选命中率", f"{stats['hit_rate']:.1%}" if stats["hit_rate"] is not None else "暂无样本")
            columns[1].metric("平均检索延迟", f"{stats['retrieval_seconds']:.3f} 秒" if stats["retrieval_seconds"] is not None else "暂无样本")
            columns[2].metric("平均 RAG 响应延迟", f"{stats['response_seconds']:.3f} 秒" if stats["response_seconds"] is not None else "暂无样本")
            st.caption(f"实际检索 {stats['attempts']} 次 · 已完成 {stats['completed']} 次 · 非空 {stats['hits']} 次 · 失败 {stats['failed']} 次。延迟统计实际检索请求，含失败；响应延迟含检索和生成，不含用户确认等待。")
            if stats["invalid_lines"]:
                st.warning(f"统计跳过 {stats['invalid_lines']} 行损坏日志，原文件保留。")
        except (OSError, ValueError) as error:
            st.warning(f"无法读取 RAG 运行指标：{error}")


show_retrieval_metrics()
def show_agent_metrics(event, show_answer=True):
    """保留最近一次请求的快照；页面重跑只重绘，不重新调用Agent。"""
    metrics = event["metrics"]
    tokens = metrics["tokens"]
    with agent_metrics_panel.container():
        if event.get("history_replayed"):
            st.caption("历史请求快照：加载保存时的公开轨迹与指标，未重新执行Agent。")
        columns = st.columns(2)
        columns[0].metric("本次 Agent Token", str(tokens["total"]) if tokens["total"] is not None else "未知")
        columns[1].metric("Agent 响应耗时", f"{metrics['response_seconds']:.3f} 秒")
        st.caption(f"阶段：{event['type']} · 请求：{event['request_id']} · 已报告 Token：{tokens['known_total']} · 用量未完整报告：{tokens['unknown_calls']} 项")
        if metrics["calls"]:
            st.dataframe([{"阶段": c["phase"], "轮次": c["iteration"], "工具": c["tool"],
                           "调用标识": ", ".join(c.get("tool_call_ids", [])) or c["id"], "输入 Token": str(c["input"]) if c["input"] is not None else "未知",
                           "输出 Token": str(c["output"]) if c["output"] is not None else "未知",
                           "用量完整": "否" if c["incomplete"] or c["input"] is None or c["output"] is None else "是"}
                          for c in metrics["calls"]], hide_index=True, width="stretch")
        for retrieval in metrics.get("retrievals", []):
            seconds = retrieval["seconds"]
            st.caption(f"本次 RAG 工具：{retrieval.get('status', '未报告')} · 返回块 {retrieval.get('returned_chunks', '未知')} · 检索耗时 "
                       + (f"{seconds:.3f} 秒" if seconds is not None else "未知"))
        tools = metrics.get("tools", {})
        columns = st.columns(2)
        rate, mean = tools.get("success_rate"), tools.get("mean_seconds")
        columns[0].metric("本次工具调用成功率", f"{rate:.1%}" if rate is not None else "暂无已返回调用")
        columns[1].metric("平均工具耗时", f"{mean:.3f} 秒" if mean is not None else "暂无耗时")
        st.caption(f"已计划 {tools.get('started', 0)} · 已返回 {tools.get('completed', 0)} · "
                   f"成功 {tools.get('successes', 0)} · 失败 {tools.get('failures', 0)} · 待返回 {tools.get('pending', 0)}。"
                   "成功指工具执行成功，不能代替任务完成或答案正确；重试按最终调用状态计一次，并行耗时可重叠。")
        if metrics.get("tool_calls"):
            st.dataframe([{"工具": c["name"], "轮次": c["iteration"], "调用ID": c["call_id"],
                           "状态": {"pending": "待返回", "success": "成功", "error": "失败"}[c["status"]],
                           "耗时（秒）": c["seconds"], "尝试次数": c["attempts"],
                           "执行方式": c["execution_mode"] or "未报告"}
                          for c in metrics["tool_calls"]], hide_index=True, width="stretch")
            st.dataframe([{"工具": c["name"], "已返回": c["completed"], "成功": c["successes"],
                           "失败": c["failures"], "待返回": c["pending"],
                           "成功率": f"{c['success_rate']:.1%}" if c["success_rate"] is not None else "暂无样本",
                           "平均耗时（秒）": c["mean_seconds"]}
                          for c in tools["by_tool"]], hide_index=True, width="stretch")
        st.markdown("**Agent 决策轨迹**")
        st.caption("展示公开决策说明与真实工具事件；分支表示同轮调用，执行方式以工具记录为准。")
        if metrics.get("trace"):
            st.graphviz_chart(trace_graph(metrics["trace"]), width="stretch")
        for iteration in dict.fromkeys(t["iteration"] for t in metrics.get("trace", [])):
            with st.expander(f"第{iteration}轮 · Thought → Action → Observation" if iteration else "请求准备与终止",
                             expanded=True):
                for step in (t for t in metrics["trace"] if t["iteration"] == iteration):
                    kind = step["type"]
                    if kind == "thought":
                        st.markdown("**Thought · 决策说明**")
                        st.write(step.get("thought", "未报告决策说明"))
                        planned = step.get("parallel_tools") or ([step["tool_name"]] if step.get("tool_name") else [])
                        st.caption(f"下一步：{step.get('next_step', '未报告')} · 计划工具：{' + '.join(planned) or '无需工具'} · 路由：{step.get('route', 'model')}")
                    elif kind == "tool_call":
                        st.markdown(f"**Action · {step['name']}**")
                        st.caption(f"调用ID：{step['call_id']}")
                        st.json(step.get("args", {}), expanded=False)
                    elif kind == "tool_result":
                        st.markdown(f"**工具返回 · {step['name']} · {'成功' if step['status'] == 'success' else '失败'}**")
                        seconds = step.get("elapsed_seconds")
                        st.caption(f"调用ID：{step['call_id']} · 执行方式：{step.get('execution_mode', '未报告')} · 耗时："
                                   + (f"{seconds:.3f} 秒" if type(seconds) in (int, float) else "未知"))
                        if step.get("error"):
                            st.warning(step["error"])
                        if step.get("pending"):
                            st.warning("等待已超时，后台工具可能仍在运行；迟到结果不会用于本次回答。")
                        st.json(step.get("result"), expanded=False)
                        if step.get("attempts"):
                            st.json({"实际尝试记录": step["attempts"]}, expanded=False)
                    elif kind == "observation":
                        st.markdown("**Observation · 结果判断**")
                        st.write(step.get("observation", "未报告观察说明"))
                        st.caption(f"决策：{step.get('decision')} · 任务完成：{step.get('task_complete')}")
                    elif kind in {"action_skipped", "recovery", "error"}:
                        st.markdown({"action_skipped": "**Action · 已跳过**", "recovery": "**错误恢复**", "error": "**阶段错误**"}[kind])
                        st.write(step.get("reason") or step.get("message", "未报告说明"))
                        if kind == "recovery":
                            st.json({"失败工具": step.get("failed_tools", []), "可选替代": step.get("available_alternatives", [])}, expanded=False)
                    elif kind == "done":
                        st.caption(f"终止原因：{step.get('stop_reason')} · 任务完成：{step.get('task_complete')}")
        if event.get("log_error"):
            st.warning(event["log_error"])
    if show_answer and event["type"] == "done":
        with agent_answer_panel.container(), st.chat_message("assistant"):
            st.markdown(event["full_response"])
            if not event["task_complete"]:
                st.warning(f"任务未完成：{event['stop_reason']}")
            show_agent_sources(event)


if agent_submitted:
    if not agent_question.strip():
        st.warning("请输入 Agent 问题。")
    else:
        st.session_state.pop("agent_last_event", None)
        with agent_user_panel.container(), st.chat_message("user"):
            st.markdown(agent_question.strip())
        try:
            for event in run_session(agent_question.strip(), st.session_state.agent_user_id,
                                     st.session_state.agent_session_id, memory=st.session_state.agent_memory, stream=True):
                st.session_state.agent_last_event = event
                if event["type"] == "token":
                    with agent_answer_panel.container(), st.chat_message("assistant"):
                        st.markdown(event["answer"])
                        st.caption("正在生成，完成状态与引用以最终校验结果为准。")
                    continue
                if event["type"] == "done":
                    st.session_state.agent_messages.append({"question": agent_question.strip(),
                        "answer": event["full_response"], "complete": event["task_complete"],
                        "stop_reason": event["stop_reason"], "event": event})
                show_agent_metrics(event)
                show_retrieval_metrics()
            if st.session_state.get("agent_last_event", {}).get("type") == "done":
                st.rerun()  # 完整问答已经入库，刷新标题与历史即可。
        except Exception as error:
            st.error(f"Agent 运行失败：{type(error).__name__}: {error}。请检查本地服务后重试。")
elif "agent_last_event" in st.session_state:
    show_agent_metrics(st.session_state.agent_last_event, show_answer=False)

st.divider()
with st.expander("知识库管理面板"):
    st.caption("共享知识库按文档内容指纹汇总，回收文档不计入活动列表。已向量化仅表示当前有索引块，"
               "不保证完整入库；导入结果/预期块数/错误仅来自当前页面批次，刷新后未知时显示“—”。")
    if library is None:
        st.error(library_error or "知识库状态读取失败，请修正后刷新。")
    else:
        summary = st.columns(3)
        summary[0].metric("知识库文档数", len(library))
        summary[1].metric("已向量化文档数", sum(document["chunks"] > 0 for document in library))
        summary[2].metric("知识库索引块数", sum(document["chunks"] for document in library))
        if not library:
            st.info("知识库暂无文档，请在左侧上传并开始导入。")
        else:
            # 磁盘/Chroma为当前状态；本页批次仅补充最近处理结果，二者不能互相冒充。
            batches = {}
            for task in st.session_state.import_tasks:
                batches.setdefault(sha256(task["data"]).hexdigest(), []).append(task)
            rows = []
            for document in library:
                batch = batches.get(document["doc_id"], [])
                # 同内容不同文件名只算一份文献，合并批次结果，不能隐藏其中一次失败。
                states = ["待索引" if task["status"] == "success" and not task.get("indexed", False)
                          else status_labels[task["status"]] for task in batch]
                rows.append({"文件名": document["name"], "文档 ID": document["doc_id"],
                             "原文状态": "已保存" if document["source_available"] else "缺失",
                             "向量化状态": "已向量化" if document["chunks"] else "未向量化",
                             "索引块数": document["chunks"], "导入结果（本页）": " / ".join(dict.fromkeys(states)) or "—",
                             "本次预期块数": " / ".join(dict.fromkeys(str(task["chunk_count"]) for task in batch
                                                                     if task.get("chunk_count"))) or "—",
                             "错误（本页）": " / ".join(dict.fromkeys(task["error"] for task in batch if task["error"])) or "—"})
            st.dataframe(rows, hide_index=True, width="stretch")

st.subheader("检索实验与服务维护")
st.subheader("文档 Top-K 检索")
st.caption("此处只检索文档块；可选向量、BM25、RRF 或 RRF + 模型重排。各类分数不可直接比较，也不是命中概率。生成答案请使用上方 RAG 问答。")
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

with st.container(border=True):
    st.markdown("**系统健康检查**")
    st.caption("按需检查本地LLM服务和现有Chroma索引，不生成回答或写入文档。状态为上次检查快照，不代表推理或检索质量。")
    if st.button("检查服务状态", key="check_health"):
        with st.spinner("正在检查本地服务与索引…"):
            st.session_state.health_result = check_health()
    if "health_result" in st.session_state:
        health = st.session_state.health_result
        st.caption(f"检查时间：{health['checked_at']} · {'两项检查正常' if health['status'] == 'ok' else '存在未就绪或异常组件'}")
        for name, key in (("LLM服务", "llm"), ("向量数据库", "vector_database")):
            component = health[key]
            text = f"{name}：{component['detail']}（检查耗时{component['seconds']:.3f}秒）"
            if component["status"] == "ok":
                st.success(text)
            elif component["status"] in {"not_initialized", "model_missing"}:
                st.warning(text)
            else:
                st.error(text)
        st.caption(f"配置模型：{health['llm'].get('model', '未报告')}")
        if health["vector_database"]["chunks"] is not None:
            st.caption(f"集合：{health['vector_database']['collection']} · 文档块数：{health['vector_database']['chunks']}")
    else:
        st.info("尚未检查；点击按钮获取当前状态。")

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
        {"模块": "一：文档处理与检索", "状态": "已实现", "范围": "批量导入、三种分块、增量索引、向量/BM25/RRF 与模型重排；五组分块及三档检索质量已实际评测"},
        {"模块": "二：RAG 生成", "状态": "已实现", "范围": "已实现 Prompt、上下文/引用、本地生成、流式、缓存、降级、请求日志与分数分布，完成参数对照；独立答案质量评测待完成"},
        {"模块": "三：Agent 决策", "状态": "已实现", "范围": "有界ReAct、八个本地工具、路由/并行/恢复、会话隔离与窗口/摘要记忆"},
        {"模块": "四：系统集成与前端", "状态": "已实现", "范围": "文档/会话管理与回收恢复、RAG/Agent正文流式、引用原文物理页、记忆、公开轨迹、实时指标与健康检查"},
        {"模块": "五：评测与交付", "状态": "人工评分待完成", "范围": "已有12篇/60题、检索与Agent实际评测、图表及Bad Case优化；基线120条和优化60条的人工质量评分待评阅"},
    ]
)
st.subheader("评阅说明")
st.write("功能测试、检索命中和模型自报完成均不能代替人工答案评分。180条实际答案按正确性、完整性和引用准确性评阅。")
