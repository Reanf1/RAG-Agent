"""文档批量导入与向量检索页面；业务功能按课程模块逐步接入。"""

import sys
from pathlib import Path

import streamlit as st


# 按文件位置导入项目模块，支持从其他目录启动 Streamlit。
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import LOADERS, create_import_tasks
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
st.info("上传文档后自动加载、分块、使用本地 M3E 批量向量化并写入 Chroma；新文档增量加入，重复块跳过。智能问答尚未接入。")

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
st.caption("搜索已持久化的知识库，返回相关文档块；可选向量、BM25 或 RRF 混合检索。各类分数不可直接比较，也不是命中概率；当前不生成答案。")
with st.form("vector_search_form"):
    method = st.selectbox("检索方式", ["向量相似度", "BM25 关键词", "RRF 混合检索"], key="retrieval_method")
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
                if method == "RRF 混合检索":
                    retriever = HybridRetriever()
                elif method == "BM25 关键词":
                    retriever = BM25Retriever()
                else:
                    retriever = VectorStore()
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
                               "RRF 混合检索": "RRF 分数"}[method]
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

st.subheader("模块开发状态")
st.table(
    [
        {"模块": "一：文档处理与检索", "状态": "部分实现", "范围": "已实现批量导入、分块、增量索引、向量/BM25/RRF 检索；模型重排待开发"},
        {"模块": "二：RAG 生成", "状态": "未实现", "范围": "引用、流式、语义缓存、降级、日志"},
        {"模块": "三：Agent 决策", "状态": "未实现", "范围": "ReAct、工具、路由、恢复、记忆"},
        {"模块": "四：系统集成与前端", "状态": "部分实现", "范围": "已有文档入库与向量搜索界面，问答、文献管理与联调待开发"},
        {"模块": "五：评测与交付", "状态": "部分实现", "范围": "已建文档骨架，评测数据和报告待完成"},
    ]
)
st.subheader("下一步")
st.write("继续实现模型重排，对 RRF 融合的候选进行精排。")
with st.sidebar:
    st.header("课程资料")
    st.write("南京农业大学生产实习课程实践")
    st.write("技术设计与验收：docs/技术设计文档.md")
