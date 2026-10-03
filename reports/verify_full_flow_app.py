"""完整流程测试启动器：只替换数据路径，运行原Streamlit页面和真实本地模型。"""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config import load_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True, help="本轮独立测试数据目录")
directory = parser.parse_args().root.resolve()
config = deepcopy(load_config())
for key, folder in (("raw_documents", "raw"), ("vector_index", "index"), ("logs", "logs")):
    config["paths"][key] = str(directory / folder)
config["paths"]["session_db"] = str(directory / "memory.sqlite3")
os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@st.cache_resource
def install_test_config(directory_string):
    """进程内只安装一次测试路径，避免并发页面的嵌套patch互相还原配置。"""
    patchers = []
    for module in ("src.utils.config", "src.agent.memory", "src.utils.logger", "src.agent.react_loop",
                   "src.agent.tools", "src.agent.router", "src.retrieval.vector_store",
                   "src.retrieval.hybrid_retriever", "src.retrieval.reranker",
                   "src.generation.rag_pipeline", "src.generation.cache", "src.chunking"):
        patcher = patch(module + ".load_config", return_value=config)
        patcher.start()
        patchers.append(patcher)
    return patchers  # 生命周期与本轮独立测试服务相同，退出进程后释放。


install_test_config(str(directory))
runpy.run_path(str(ROOT / "src/frontend/app.py"), run_name="__main__")
