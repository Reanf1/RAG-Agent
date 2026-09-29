"""批量导入、文档检索与本地 RAG 流式问答页面。"""

import sys
from pathlib import Path
from time import perf_counter

import streamlit as st


# 按文件位置导入项目模块，支持从其他目录启动 Streamlit。
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import LOADERS, create_import_tasks
from src.generation.rag_pipeline import build_context
from src.generation.streaming import stream_answer
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
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

st.subheader("RAG 流式问答")
st.caption("每个问题独立检索 Top-K，再由本地 Ollama 生成；当前历史仅保留页面展示，不作为多轮推理上下文。引用对应实际文件和位置，原文证据可展开查看。")
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []
if st.button("清空当前对话", key="clear_rag_chat"):
    st.session_state.rag_messages = []
    st.rerun()


def show_answer_details(message):
    """历史与本轮共用完成状态、实际用量和原文证据展示。"""
    if message.get("error"):
        st.error(f"回答未完成：{message['error']}。已保留部分文本；请检查本地服务或配置后重新提交问题。")
    elif message.get("complete"):
        usage = message["usage"]
        st.caption(f"服务已结束 · 检索 {message['retrieval_seconds']:.2f} 秒 · "
                   f"总耗时 {message['elapsed_seconds']:.2f} 秒 · "
                   f"输入 Token {usage['prompt_eval_count']} · 输出 Token {usage['eval_count']}")
    for warning in message.get("warnings", []):
        st.warning(warning)
    for reference in message["citations"]:
        with st.expander(f"参考文档{reference['id']} · {reference['source_file'] or '来源信息未提供'} · {reference['location']}"):
            st.caption(f"块 ID：{reference['metadata'].get('chunk_id', '')}；仅展示本轮送入模型的原文证据。")
            st.text(reference["text"])


for message in st.session_state.rag_messages:
    with st.chat_message("user"):
        st.write(message["question"])
    with st.chat_message("assistant"):
        st.markdown(message["answer"])
        show_answer_details(message)

question = st.chat_input("询问已上传论文（支持中英文）", key="rag_question")
if question and question.strip():
    message = {"question": question, "answer": "", "citations": [], "warnings": [], "complete": False}
    started = perf_counter()
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        try:
            with st.spinner("正在检索本地文献…"):
                results = HybridRetriever().search(question, k=config["retrieval"]["top_k"], rerank=True)
                context = build_context(question, results)
            message["retrieval_seconds"] = perf_counter() - started
            if not results:
                st.info("当前知识库中未找到相关文档；以下回答没有文献依据。")
            answer_placeholder.markdown("正在等待本地模型输出…")
            # 采用完整快照替换占位区，引用闭合后立即补全；不是先等全文再模拟打字。
            for event in stream_answer(question, context):
                if event["type"] == "token":
                    message["answer"], message["citations"] = event["answer"], event["citations"]
                    answer_placeholder.markdown(event["answer"] + " ▌")
                else:
                    message.update(event)
                    message["complete"] = event["type"] == "done"
            if not message["complete"] and not message.get("message"):
                raise RuntimeError("生成流未返回完成标记")
            if message.get("type") == "error":
                message["error"] = message["message"]
        except Exception as error:
            message["error"] = f"{type(error).__name__}: {error}"
        message["elapsed_seconds"] = perf_counter() - started
        answer_placeholder.markdown(message["answer"])
        show_answer_details(message)
    st.session_state.rag_messages.append(message)

st.subheader("模块开发状态")
st.table(
    [
        {"模块": "一：文档处理与检索", "状态": "部分实现", "范围": "已实现批量导入、分块、增量索引、向量/BM25/RRF 与模型重排；三档质量已评测，分块召回对比待完成"},
        {"模块": "二：RAG 生成", "状态": "部分实现", "范围": "已实现 Prompt、上下文/引用、本地生成及流式页面，完成参数对照；缓存、完整降级、日志待完成"},
        {"模块": "三：Agent 决策", "状态": "未实现", "范围": "ReAct、工具、路由、恢复、记忆"},
        {"模块": "四：系统集成与前端", "状态": "部分实现", "范围": "已有文档入库、检索与 RAG 流式问答；Agent、持久会话、文献管理与完整联调待开发"},
        {"模块": "五：评测与交付", "状态": "部分实现", "范围": "已有三档检索实测、图表与 Excel；完整系统评测待完成"},
    ]
)
st.subheader("下一步")
st.write("后续继续语义缓存、完整降级与日志；分块召回对比和 Agent 等课程要求仍保留。")
with st.sidebar:
    st.header("课程资料")
    st.write("南京农业大学生产实习课程实践")
    st.write("技术设计与验收：docs/技术设计文档.md")
