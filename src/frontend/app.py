"""项目初始化页面；明确展示开发状态，业务功能按课程模块逐步接入。"""

from pathlib import Path

import streamlit as st
import yaml


# 按入口文件定位根目录，使配置读取不依赖启动时的工作目录。
project_root = Path(__file__).resolve().parents[2]
with (project_root / "config.yaml").open(encoding="utf-8") as config_file:
    app_config = yaml.safe_load(config_file)["app"]

st.set_page_config(page_title=app_config["name"], layout="wide")
st.title(app_config["name"])
st.caption(app_config["description"])
st.info("项目骨架已初始化。文档上传、知识库检索和智能问答将按课程模块逐步实现。")

# 仅展示真实状态，不提供尚未接入后端的上传或问答控件。
st.subheader("模块开发状态")
st.table(
    [
        {"模块": "一：文档处理与检索", "状态": "未实现", "范围": "加载、分块、向量、BM25、RRF、重排序"},
        {"模块": "二：RAG 生成", "状态": "未实现", "范围": "引用、流式、语义缓存、降级、日志"},
        {"模块": "三：Agent 决策", "状态": "未实现", "范围": "ReAct、工具、路由、恢复、记忆"},
        {"模块": "四：系统集成与前端", "状态": "部分实现", "范围": "已建启动页，业务界面和联调待开发"},
        {"模块": "五：评测与交付", "状态": "部分实现", "范围": "已建文档骨架，评测数据和报告待完成"},
    ]
)
st.subheader("下一步")
st.write("先完成文档加载与来源元数据，再进行分块策略和检索质量对比实验。")
with st.sidebar:
    st.header("课程资料")
    st.write("南京农业大学生产实习课程实践")
    st.write("技术设计与验收：docs/技术设计文档.md")
