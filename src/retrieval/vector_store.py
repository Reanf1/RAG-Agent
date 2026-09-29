"""本地 Embedding 接口；持久化索引、增量更新与 Top-K 检索待实现。"""

import os
from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import load_config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """首次调用才加载唯一配置的本地模型；加载失败直接报错，不转云端。"""
    project_root = Path(__file__).resolve().parents[2]
    config = load_config()["embedding"]
    model_path = project_root / config["local_path"]
    if not model_path.is_dir():
        raise FileNotFoundError(f"本地 Embedding 模型不存在：{model_path}，请先按用户手册下载权重")
    # 缓存留在项目模型目录；不在页面启动或模型编码时自动下载权重。
    os.environ.setdefault("HF_HOME", str(project_root / "data/models/.hf-runtime"))
    return HuggingFaceEmbeddings(
        model_name=str(model_path),
        model_kwargs={"device": config["device"], "local_files_only": True, "trust_remote_code": False},
        encode_kwargs={"normalize_embeddings": True, "batch_size": config["batch_size"]},
    )
