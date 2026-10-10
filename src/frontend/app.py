"""科研多轮对话、文档检索与知识库管理页面。

Streamlit交互会从头重跑本文件；已完成消息、上传进度和待确认候选保存在
session_state中。只有提交上传或问题的分支执行业务请求，普通重绘读取
已有快照。知识库跨会话共享，聊天记忆及RAG缓存按用户/会话隔离。
"""

import sys
import json
import re
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import streamlit as st


# 按文件位置导入项目模块，支持从其他目录启动 Streamlit。
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import LOADERS, create_import_tasks
from src.frontend.components.documents import list_documents, delete_document, restore_document, read_pdf_page, read_document_content
from src.frontend.components.trace import execution_rows, conversation_statistics, record_runtime_success
from src.agent import run_session
from src.agent.tools import get_available_tools
from src.generation.cache import SemanticCache
from src.frontend.components.sessions import render_sessions
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.logger import request_time, retrieval_score_distribution
from src.utils.messages import document_location
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
    shown_pages = set()
    for item in event.get("context", {}).get("observations", []):
        result = item.get("result")
        for reference in result.get("citations", []) if isinstance(result, dict) else []:
            if Path(reference["source_file"]).suffix.lower() == ".pdf" and "page_number" in reference["metadata"]:
                # 不合并引用正文，只将同一内容ID、物理页的原文入口展示一次。
                metadata = reference["metadata"]
                identity = metadata.get("doc_id") or (item.get("call_id"), reference["source_file"])
                page_key = (identity, metadata["page_number"])
                if page_key in shown_pages:
                    continue
                shown_pages.add(page_key)
                key = f"agent-citation:{event['request_id']}:{item.get('call_id')}:{reference['id']}"
                if st.button(f"查看{reference['source_file']} · {reference['location']}", key=key):
                    show_original_page(reference)
        if isinstance(result, dict) and result.get("status") == "needs_confirmation":
            with st.expander("低相关候选：请核对原文", expanded=True):
                for reference in result.get("references", []):
                    st.caption(f"{reference['source_file']} · {reference['location']} · 重排分数 {reference['score']:.4f}")
                    st.markdown(reference["text"])
                    if Path(reference["source_file"]).suffix.lower() == ".pdf" and "page_number" in reference["metadata"]:
                        if st.button(f"查看{reference['source_file']} · {reference['location']}",
                                     key=f"agent-candidate:{event['request_id']}:{item.get('call_id')}:{reference['id']}"):
                            show_original_page(reference)
                identifier = result.get("confirmation_id")
                if identifier in st.session_state.agent_pending_rag:
                    confirm, cancel = st.columns(2)
                    if confirm.button("确认使用候选", key=f"confirm-rag:{identifier}"):
                        st.session_state.agent_confirmed_rag = st.session_state.agent_pending_rag.pop(identifier)
                        st.rerun()
                    if cancel.button("取消", key=f"cancel-rag:{identifier}"):
                        st.session_state.agent_pending_rag.pop(identifier)
                        st.rerun()


raw_dir = project_root / config["paths"]["raw_documents"]
index_dir = project_root / config["paths"]["vector_index"]
max_file_size_mb = config["importing"]["max_file_size_mb"]

st.set_page_config(page_title=app_config["name"], layout="wide")
st.title(app_config["name"])
st.caption(app_config["description"])

# 导入状态仅存于当前页面会话，原始文件成功加载后保存到本地。
if "import_tasks" not in st.session_state:
    st.session_state.import_tasks = []
if "import_progress" not in st.session_state:
    st.session_state.import_progress = {"completed": 0, "total": 0}
if "upload_version" not in st.session_state:
    st.session_state.upload_version = 0

with st.sidebar:
    st.header("系统状态")
    st.caption(f"系统时间：{datetime.fromisoformat(request_time()).strftime('%Y-%m-%d %H:%M:%S')}")
    runtime_owner = (config["llm"]["base_url"], config["llm"]["model"], str(index_dir), config["retrieval"]["collection_name"])
    if st.session_state.get("runtime_owner") != runtime_owner:
        st.session_state.pop("health_result", None)
        st.session_state.runtime_owner = runtime_owner
    # 首次或配置／索引改变时检查；普通交互复用快照，不启动模型推理。
    if "health_result" not in st.session_state:
        st.session_state.health_result = check_health()
        st.session_state.runtime_checks = {}
    health = st.session_state.health_result
    runtime_captions = {}
    for name, key, selected in (("LLM服务", "llm", config["llm"]["model"]),
                                ("向量数据库", "vector_database", "Chroma")):
        component = health[key]
        if component["status"] == "ok":
            st.success(f"{name}：{'可连接，模型已安装' if key == 'llm' else '索引可读取'}（{component.get('model', selected)}）")
        else:
            text = f"{name}：异常（{component['detail']}）"
            if component["status"] in {"not_initialized", "model_missing"}:
                st.warning(text)
            else:
                st.error(text)
        action = "推理" if key == "llm" else "向量检索"
        tested_at = st.session_state.runtime_checks.get(key)
        runtime_captions[key] = st.empty()
        runtime_captions[key].caption(f"最近实际{action}成功：{tested_at}" if tested_at else f"本页尚无实际{action}记录。")
    st.caption("连接检查与最近业务记录分别展示；成功执行不代表答案质量已通过审核。")
    if st.button("刷新状态", key="check_health"):
        st.session_state.health_result = check_health()
        st.rerun()
    st.header("文档上传与管理")
    uploaded_files = st.file_uploader(
        "上传文档", type=[suffix.lstrip(".") for suffix in LOADERS],
        accept_multiple_files=True, max_upload_size=max_file_size_mb,
        key=f"document_uploads:{st.session_state.upload_version}",
    )
    st.caption(f"支持 PDF、Word、TXT/Markdown，单份最大 {max_file_size_mb} MB。")
    selected_tasks = create_import_tasks([(file.name, file.getvalue()) for file in uploaded_files])
    same_selection = [t["id"] for t in selected_tasks] == [t["id"] for t in st.session_state.import_tasks]
    unfinished = any(t["status"] in {"pending", "loading", "chunking", "indexing"}
                     or (t["status"] == "success" and not t.get("indexed", False))
                     for t in st.session_state.import_tasks)
    # 横向容器不会像columns在窄窗口自动叠成两行。
    with st.container(horizontal=True, gap="small"):
        start_import = st.button("开始导入", key="start_import",
                                 disabled=not selected_tasks or (same_selection and not unfinished))
        retry_import = st.button(
            "重试失败项", key="retry_import",
            disabled=not any(t["status"] == "failed" for t in st.session_state.import_tasks),
        )

with st.sidebar:
    import_status = st.empty()


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
        st.subheader("导入状态")
        st.progress(progress["completed"] / total,
                    text=f"本次操作已处理 {progress['completed']}/{total} 份")
        if finished:
            st.caption("累计导入结果")
            st.dataframe([{"文件": t["name"], "状态": "成功" if t["status"] == "success" else "失败"}
                          for t in finished], hide_index=True, width="stretch")
        successes = sum(t["status"] == "success" for t in finished)
        failures = len(finished) - successes
        st.caption(f"累计成功 {successes} 份，失败 {failures} 份。")
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
    st.session_state.pop("health_result", None)
    st.rerun()

show_import_status()


session_ready = render_sessions(project_root / config["paths"]["session_db"])

# 统一科研对话入口，Agent按需要调用模块二RAG工具。
chat_tab, retrieval_tab, knowledge_tab = st.tabs(["科研对话", "文档检索", "知识库"])
library = None  # 读取失败保持未知，不能显示为正常空库。
library_error = ""
with knowledge_tab:
    document_list, document_content = st.columns([2, 3], gap="medium")
    with document_list:
        st.subheader("知识库文档")
        # 四字确认文字不换行，按钮保持原“删除”的54×40像素尺寸。
        st.html("""<style>
            [class*="st-key-delete_document-"] button,
            .st-key-cancel_delete_document button,
            .st-key-confirm_delete_document button {padding: 0 1px; height: 40px;}
            [class*="st-key-delete_document-"] button p {white-space: nowrap;}
            [class*="st-key-knowledge_document-"] button div[title],
            [class*="st-key-knowledge_document-"] button p {white-space: normal; overflow-wrap: anywhere;}
            .st-key-confirm_delete_document button p {font-size: 12px; white-space: nowrap;}
        </style>""")
        if "document_notice" in st.session_state:
            st.info(st.session_state.pop("document_notice"))
        try:
            try:
                library = list_documents(raw_dir, index_dir)
            except Exception:
                # 库读取失败时仍保留刷新入口，便于修复后重新读取。
                st.button("刷新知识库状态", key="refresh_knowledge")
                raise
            identifiers = [document["doc_id"] for document in library]
            if st.session_state.get("knowledge_document_id") not in identifiers:
                st.session_state.knowledge_document_id = identifiers[0] if identifiers else None
            confirm_delete = cancel_delete = False
            # 文件列表与右侧原文区同高，长列表只在框内滚动。
            with st.container(height=600, border=True, key="knowledge_document_list"):
                if not library:
                    st.caption("知识库暂无文档。")
                for document in library:
                    pending = st.session_state.get("delete_pending") == document["doc_id"]
                    # 为操作列留出空间，文件名换行显示，按钮保持54×40像素。
                    details, actions = st.columns([2, 1], gap="small", vertical_alignment="center")
                    with details:
                        selected = st.session_state.knowledge_document_id == document["doc_id"]
                        if st.button(document["name"], key=f"knowledge_document:{document['doc_id']}",
                                     type="primary" if selected else "secondary", width="stretch"):
                            st.session_state.knowledge_document_id = document["doc_id"]
                            st.rerun()
                    with actions:
                        if pending:
                            confirm_delete = st.button("确认删除", key="confirm_delete_document", width=54)
                            cancel_delete = st.button("取消", key="cancel_delete_document", width=54)
                        elif st.button("删除", key=f"delete_document:{document['doc_id']}",
                                     help=f"删除{document['name']}", disabled=not document["source_available"], width=54):
                            st.session_state.delete_pending = document["doc_id"]
                            st.rerun()
                    st.caption(f"ID：{document['doc_id'][:8]}")
            # 刷新放在文档列表下方、已归档知识上方。
            st.button("刷新知识库状态", key="refresh_knowledge")
            if library:
                if "delete_pending" in st.session_state:
                    pending_id = st.session_state.delete_pending
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
                        st.session_state.pop("delete_pending")
                        st.session_state.pop("health_result", None)
                        st.session_state.document_notice = f"已删除 {removed} 个检索块，原文已回收，可在下方恢复。"
                        st.rerun()
            trash_dir = raw_dir / ".trash"
            archived = sorted(folder.name for folder in trash_dir.iterdir()
                              if folder.is_dir() and not folder.is_symlink() and len(folder.name) == 64
                              and all(c in "0123456789abcdef" for c in folder.name)) if trash_dir.is_dir() and not trash_dir.is_symlink() else []
            if archived:
                archive_names = {identifier: " / ".join(f.name for f in sorted((trash_dir / identifier).iterdir())
                                 if f.is_file() and not f.is_symlink() and f.suffix.lower() in LOADERS)
                                 for identifier in archived}
                # 单选框只有选择操作，名称和ID不作为可编辑文本输入。
                restore_id = st.radio("已归档知识", archived, key="restore_doc_id", width="stretch",
                                     format_func=lambda identifier: archive_names[identifier],
                                     captions=[f"ID：{identifier[:8]}" for identifier in archived])
                if st.button("恢复", key="restore_document"):
                    st.session_state.import_tasks = restore_document(raw_dir, restore_id)
                    for progress in batch_build_index(st.session_state.import_tasks, raw_dir, max_file_size_mb):
                        st.session_state.import_progress = progress
                        show_import_status()
                    st.session_state.knowledge_document_id = restore_id
                    st.session_state.document_notice = "原文已恢复，请查看本批导入状态；失败项可重试。"
                    st.session_state.pop("health_result", None)
                    st.rerun()
        except Exception as error:
            library_error = f"文档管理失败：{type(error).__name__}: {error}。原文保留，请修正后重试。"
            st.error(library_error)
    with document_content:
        selected_document = next((document for document in library or []
                                  if document["doc_id"] == st.session_state.get("knowledge_document_id")), None)
        if selected_document:
            st.subheader(selected_document["name"])
            st.caption(f"{selected_document['chunks']} 个索引块 · "
                       + selected_document["index_status"])
            try:
                with st.container(height=600, border=True, key="knowledge_content"):
                    for part in read_document_content(raw_dir, selected_document["doc_id"]):
                        metadata = part.metadata
                        if "page_number" in metadata:
                            end = metadata.get("page_end", metadata["page_number"])
                            pages = str(metadata["page_number"]) if end == metadata["page_number"] else f"{metadata['page_number']}–{end}"
                            st.caption(f"原文第 {pages} 页" + (" · 表格" if metadata.get("content_type") == "table" else ""))
                        elif "table_index" in metadata:
                            st.caption(f"原文表格 {metadata['table_index']}")
                        st.markdown(part.page_content)
            except Exception as error:
                st.error(f"无法读取原文：{type(error).__name__}: {error}")
        elif library is not None:
            st.info("请在左侧上传文档，导入后在此选择文件查看内容。")


def show_turn_footer(message):
    """每轮消息附真实最终用量与耗时，旧历史缺失指标时明确显示未知。"""
    metrics = message.get("event", {}).get("metrics", {})
    tokens = metrics.get("tokens", {}).get("total")
    seconds = metrics.get("response_seconds")
    usage = str(tokens) if tokens is not None else "未知"
    elapsed = f"{seconds:.3f} 秒" if seconds is not None else "未知"
    st.caption(f"本次 Token：{usage} · Agent响应耗时：{elapsed}")
    if message.get("complete") is False:
        st.warning(f"任务未完成：{message.get('stop_reason', '未知')}")
    if message.get("error") or message.get("event", {}).get("error"):
        st.error(message.get("error") or message["event"]["error"])
    if message.get("event", {}).get("log_error"):
        st.warning(message["event"]["log_error"])
    for item in message.get("event", {}).get("context", {}).get("observations", []):
        payload = item.get("result")
        if isinstance(payload, dict) and payload.get("cache", {}).get("hit"):
            st.caption(f"RAG缓存命中：{payload['cache']['mode']} · 已跳过本次检索和RAG生成")
    show_agent_sources(message.get("event", {}))


def show_execution_table(event, question, expanded=True):
    """保留逐轮七列统计，并展示已有公开计划、参数、结果与观察。"""
    rows = execution_rows(event)
    with st.expander(question[:60], expanded=expanded):
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.caption("该轮尚未记录阶段指标。")
        if event.get("metrics", {}).get("trace"):
            with st.popover("查看公开执行过程"):
                st.caption("工具计划、参数、实际结果和完成判断；来自已有事件。")
                # 只缩短展示副本中的长指纹；真实参数、日志和会话保持完整ID。
                display = re.sub(r"(?<![0-9a-f])[0-9a-f]{32,64}(?![0-9a-f])",
                                 lambda match: match.group()[:8], json.dumps(event["metrics"]["trace"], ensure_ascii=False))
                st.json(display)


def show_statistics():
    """统计当前会话全部请求；失败工具和未返回工具分别计数。"""
    stats = conversation_statistics(st.session_state.agent_messages)
    st.subheader("状态统计")
    st.caption("当前会话累计；工具成功率只统计已返回调用，未报告的用量不计为零。")
    columns = st.columns(4)
    columns[0].metric("对话次数", stats["requests"])
    columns[1].metric("累计Token", str(stats["tokens"]) if stats["tokens"] is not None else "未知")
    columns[2].metric("累计响应耗时", f"{stats['seconds']:.3f} 秒" if stats["seconds"] is not None and stats["requests"] else ("未知" if stats["requests"] else "暂无样本"))
    columns[3].metric("平均响应耗时", f"{stats['mean_seconds']:.3f} 秒" if stats["mean_seconds"] is not None else "暂无样本")
    columns = st.columns(4)
    columns[0].metric("任务完成率", f"{stats['successes'] / stats['completed_requests']:.1%}" if stats["completed_requests"] else "暂无样本")
    columns[1].metric("工具调用次数", stats["tool_calls"])
    columns[2].metric("工具调用成功率", f"{stats['tool_rate']:.1%}" if stats["tool_rate"] is not None else "暂无已返回调用")
    columns[3].metric("平均推理轮次", f"{stats['mean_rounds']:.2f}" if stats["mean_rounds"] is not None else "未知")
    hit_rate = f"{stats['retrieval_hit_rate']:.1%}" if stats["retrieval_hit_rate"] is not None else "未知"
    retrieval_seconds = f"{stats['mean_retrieval_seconds']:.3f} 秒" if stats["mean_retrieval_seconds"] is not None else "未知"
    tool_seconds = f"{stats['mean_tool_seconds']:.3f} 秒" if stats["mean_tool_seconds"] is not None else "未知"
    st.caption(f"RAG检索 {stats['retrieval_count']} 次 · 候选命中率 {hit_rate} · 平均检索耗时 {retrieval_seconds} · 平均工具耗时 {tool_seconds}。")
    st.caption(f"工具成功 {stats['tool_successes']} · 失败 {stats['tool_failures']} · 待返回 {stats['tool_pending']}；"
               f"已报告Token {stats['known_tokens']} · 用量未知 {stats['unknown_tokens']} 次。")


with chat_tab:
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    # 缓存仅属于当前用户/会话；切换会话或配置后重新创建，不缓存整个Agent对话。
    cache_owner = (st.session_state.get("agent_user_id"), st.session_state.get("agent_session_id"), config["generation"]["cache"])
    if st.session_state.get("agent_cache_owner") != cache_owner:
        st.session_state.agent_cache_owner = deepcopy(cache_owner)
        st.session_state.agent_rag_cache = SemanticCache()
        st.session_state.agent_pending_rag = {}
        st.session_state.pop("agent_confirmed_rag", None)
    # 固定高度消息区内部滚动；新增消息和流式增量自动滚到最新内容。
    history_panel = st.container(height=460, border=True, key="agent_chat_history", autoscroll=True)
    with history_panel:
        for index, previous in enumerate(st.session_state.agent_messages):
            with st.container(border=True, key=f"agent_turn_{index}"):
                with st.chat_message("user"):
                    st.markdown(previous["question"])
                with st.chat_message("assistant"):
                    st.markdown(previous["answer"])
                    show_turn_footer(previous)
    agent_question = st.chat_input("输入消息…", key=f"agent_question:{st.session_state.get('agent_session_id', 'unavailable')}", disabled=not session_ready)
    approval = st.session_state.pop("agent_confirmed_rag", None) if session_ready else None
    if approval:
        agent_question = approval["question"]
    execution_panel = st.empty()
    statistics_panel = st.empty()
    if agent_question and agent_question.strip():
        st.session_state.agent_pending_rag.clear()  # 新问题使此前候选确认失效；确认快照已单独取出。
        question = agent_question.strip()
        incoming = {"question": question, "answer": "", "complete": False, "event": {}}
        started = perf_counter()
        with history_panel, st.container(border=True):
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                answer_panel = st.empty()
                with st.spinner("正在处理…"):
                    try:
                        for event in run_session(question, st.session_state.agent_user_id,
                                                 st.session_state.agent_session_id,
                                                 tools=get_available_tools(cache=st.session_state.agent_rag_cache,
                                                                           session_id=st.session_state.agent_session_id,
                                                                           pending=st.session_state.agent_pending_rag,
                                                                           confirmation=approval, request_question=question),
                                                 confirmed_rag_args=(approval["args"] if "args" in approval else
                                                                     {"question": approval["tool_question"], "doc_id": approval["doc_id"]}) if approval else None,
                                                 memory=st.session_state.agent_memory, stream=True):
                            record_runtime_success(st.session_state.runtime_checks, event,
                                                   config["llm"]["model"], request_time())
                            if event["type"] == "token":
                                incoming["answer"] = event["answer"]
                                answer_panel.markdown(incoming["answer"] + " ▌")
                                continue
                            incoming["event"] = event
                            with execution_panel.container():
                                show_execution_table(event, question)
                            if event["type"] == "done":
                                incoming.update(answer=event["full_response"], complete=event["task_complete"],
                                                stop_reason=event["stop_reason"])
                                answer_panel.markdown(incoming["answer"])
                        if incoming["event"].get("type") != "done":
                            raise RuntimeError("生成流未返回完成标记，已保留部分文本")
                    except Exception as error:
                        incoming.update(error=f"Agent运行失败：{type(error).__name__}: {error}。请检查本地服务后重试。",
                                        stop_reason="stream_error", complete=False)
                        failure = deepcopy(incoming["event"])
                        failure.update(type="done", task_complete=False, stop_reason="stream_error", error=incoming["error"])
                        failure.setdefault("metrics", {})["response_seconds"] = perf_counter() - started
                        incoming["event"] = failure
                        # 异常流没有由run_session保存完整轮次，显式保留问题与部分回答。
                        try:
                            st.session_state.agent_memory.append_turn(st.session_state.agent_user_id,
                                st.session_state.agent_session_id, question, incoming["answer"] or incoming["error"],
                                details={"task_complete": False, "stop_reason": "stream_error", "event": failure})
                        except Exception as save_error:
                            incoming["error"] += f"；历史保存失败：{save_error}"
                answer_panel.markdown(incoming["answer"] or incoming.get("error", ""))
                show_turn_footer(incoming)
        st.session_state.agent_messages.append(incoming)
        st.session_state.agent_last_event = incoming["event"]
        st.rerun()  # 重绘已完成消息、侧栏标题和累计统计，不再次请求模型。
    with execution_panel.container():
        for index, message in enumerate(st.session_state.agent_messages):
            show_execution_table(message.get("event", {}), message["question"],
                                 expanded=index == len(st.session_state.agent_messages) - 1)
    with statistics_panel.container():
        show_statistics()

with knowledge_tab:
    st.subheader("知识库管理面板")
    if library is None:
        st.error(library_error or "知识库状态读取失败，请修正后刷新。")
    else:
        summary = st.columns(3)
        summary[0].metric("知识库文档数", len(library))
        summary[1].metric("已向量化文档数", sum(document["index_status"] == "已向量化" for document in library))
        summary[2].metric("知识库索引块数", sum(document["chunks"] for document in library))
        if not library:
            st.info("知识库暂无文档，请在左侧上传并开始导入。")
        else:
            # 表格仅显示磁盘与Chroma的当前状态，批次失败仍在左侧导入区查看。
            rows = [{"文件名": document["name"], "文档 ID": document["doc_id"][:8],
                     "原文状态": "已保存" if document["source_available"] else "缺失",
                     "向量化状态": document["index_status"],
                     "索引块数": document["chunks"]} for document in library]
            st.dataframe(rows, hide_index=True, width="stretch")

with retrieval_tab:
    st.subheader("文档 Top-K 检索")
    with st.form("vector_search_form"):
        method_column, count_column, document_column = st.columns([2, 1, 2])
        method = method_column.selectbox("检索方式", ["向量相似度", "BM25 关键词", "RRF 混合检索", "RRF + 模型重排"], key="retrieval_method")
        top_k = count_column.number_input("Top-k", min_value=1,
                                          value=config["retrieval"]["top_k"], step=1, key="vector_top_k")
        doc_id = document_column.text_input("文档ID", key="vector_doc_id", help="留空检索全部文献；支持前8位ID。")
        query = st.text_input("查询内容", key="vector_query")
        search_submitted = st.form_submit_button("检索", key="vector_search")

    if search_submitted:
        if not query.strip():
            st.warning("请输入查询内容。")
        else:
            try:
                # 简写仅用于界面输入；唯一匹配后以完整ID查询，冲突时明确提示。
                selected_id = doc_id.strip() or None
                if selected_id and len(selected_id) == 8:
                    if library is None:
                        raise ValueError("知识库状态读取失败，无法匹配简写文档ID")
                    matches = [document["doc_id"] for document in library if document["doc_id"].startswith(selected_id)]
                    if len(matches) > 1:
                        raise ValueError("前8位ID对应多份文档，请使用完整ID")
                    if matches:
                        selected_id = matches[0]
                # BM25 每次提交从当前正文重建小规模内存索引，无需加载 M3E。
                with st.spinner("正在检索本地知识库…"):
                    if method in ("RRF 混合检索", "RRF + 模型重排"):
                        retriever = HybridRetriever()
                    elif method == "BM25 关键词":
                        retriever = BM25Retriever()
                    else:
                        retriever = VectorStore()
                    if method == "RRF + 模型重排":
                        results = retriever.search(query, k=top_k, doc_id=selected_id, rerank=True)
                    else:
                        results = retriever.search(query, k=top_k, doc_id=selected_id)
                    if method != "BM25 关键词":
                        st.session_state.runtime_checks["vector_database"] = request_time()
                        # 侧栏先于检索表单绘制，原位更新才能在本次提交立即显示成功时间。
                        runtime_captions["vector_database"].caption(
                            f"最近实际向量检索成功：{st.session_state.runtime_checks['vector_database']}")
            except Exception as error:
                st.error(f"检索失败：{type(error).__name__}: {error}。请根据错误信息检查配置后重新检索。")
            else:
                if not results:
                    st.info("没有可检索的文档块或关键词无匹配，请先导入文档并检查关键词；如填写了文档 ID，请检查是否正确。")
                st.caption(f"返回 {len(results)} 个文档块（Top-K={top_k}）。")
                for notice in dict.fromkeys(document.metadata["retrieval_warning"] for document, _ in results
                                            if document.metadata.get("retrieval_warning")):
                    st.warning(notice)
                for rank, (document, score) in enumerate(results, 1):
                    metadata = document.metadata
                    filename = metadata.get("source_file", "未知文件")
                    score_label = {"向量相似度": "余弦相似度", "BM25 关键词": "BM25 分数",
                                   "RRF 混合检索": "RRF 分数", "RRF + 模型重排": "模型相关性分数"}[method]
                    with st.expander(f"{rank}. {filename} · {score_label} {score:.4f}", expanded=True):
                        st.caption(f"来源：{filename}；{document_location(metadata)}")
                        st.caption(f"文档 ID：{metadata.get('doc_id', '')[:8]}；块 ID：{metadata.get('chunk_id', '')[:8]}")
                        st.text(document.page_content)

    st.subheader("检索分数分布")
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
