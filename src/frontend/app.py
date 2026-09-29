"""文档批量导入页面；业务功能按课程模块逐步接入。"""

import sys
from pathlib import Path

import streamlit as st


# 按文件位置导入项目模块，支持从其他目录启动 Streamlit。
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import LOADERS, batch_import, create_import_tasks
from src.utils.config import load_config

config = load_config()
app_config = config["app"]
raw_dir = project_root / config["paths"]["raw_documents"]
max_file_size_mb = config["importing"]["max_file_size_mb"]

st.set_page_config(page_title=app_config["name"], layout="wide")
st.title(app_config["name"])
st.caption(app_config["description"])
st.info("当前支持文档保存与正文加载；分块、本地 Embedding 和 Chroma 索引可通过 Python 调用，导入流程尚未接入分块、索引和智能问答。")

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
    unfinished = any(t["status"] in {"pending", "loading"} for t in st.session_state.import_tasks)
    start_import = st.button("开始导入", key="start_import",
                             disabled=not selected_tasks or (same_selection and not unfinished))
    retry_import = st.button(
        "重试失败项", key="retry_import",
        disabled=not any(t["status"] == "failed" for t in st.session_state.import_tasks),
    )

st.subheader("本批导入状态")
progress_bar = st.progress(0)
status_table = st.empty()
status_labels = {"pending": "等待", "loading": "处理中", "success": "成功", "failed": "失败"}


def show_import_status():
    """刷新同一处进度和表格，不将内容单元误标为向量分块。"""
    progress = st.session_state.import_progress
    total = progress["total"]
    progress_bar.progress(progress["completed"] / total if total else 0,
                          text=f"已处理 {progress['completed']}/{total} 份")
    tasks = st.session_state.import_tasks
    if tasks:
        status_table.dataframe([{
            "文件": t["name"], "状态": status_labels[t["status"]],
            "尝试次数": t["attempts"], "内容单元": len(t["documents"]),
            "耗时（秒）": t["elapsed"], "错误": t["error"],
        } for t in tasks], hide_index=True, width="stretch")


if start_import and not same_selection:
    st.session_state.import_tasks = selected_tasks
if start_import or retry_import:
    for progress in batch_import(st.session_state.import_tasks, raw_dir,
                                 max_file_size_mb, retry_failed=retry_import):
        st.session_state.import_progress = progress
        show_import_status()
    # 完成后刷新按钮禁用状态，避免页面重跑或重复点击再次导入成功项。
    st.rerun()

show_import_status()
tasks = st.session_state.import_tasks
if tasks:
    successes = sum(t["status"] == "success" for t in tasks)
    failures = sum(t["status"] == "failed" for t in tasks)
    st.caption(f"加载成功 {successes} 份，失败 {failures} 份。进度表示处理完成数量。")
    if failures:
        st.warning("请根据错误信息修正文件后重新上传，或点击“重试失败项”再尝试一轮。")
else:
    st.caption("在左侧选择文档后，点击“开始导入”。")

st.subheader("模块开发状态")
st.table(
    [
        {"模块": "一：文档处理与检索", "状态": "部分实现", "范围": "已实现批量导入、分块、M3E 和 Chroma；混合检索与重排待开发"},
        {"模块": "二：RAG 生成", "状态": "未实现", "范围": "引用、流式、语义缓存、降级、日志"},
        {"模块": "三：Agent 决策", "状态": "未实现", "范围": "ReAct、工具、路由、恢复、记忆"},
        {"模块": "四：系统集成与前端", "状态": "部分实现", "范围": "已有文档导入界面，问答、文献管理与联调待开发"},
        {"模块": "五：评测与交付", "状态": "部分实现", "范围": "已建文档骨架，评测数据和报告待完成"},
    ]
)
st.subheader("下一步")
st.write("将已实现的分块与 Chroma 索引接入导入页面，继续实现混合检索与重排。")
with st.sidebar:
    st.header("课程资料")
    st.write("南京农业大学生产实习课程实践")
    st.write("技术设计与验收：docs/技术设计文档.md")
