"""读取唯一YAML配置，按需检查本地模型服务和向量数据库。"""

from datetime import datetime
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx
import yaml


def load_config() -> dict:
    """按模块位置定位项目根目录，返回配置字典。"""
    path = Path(__file__).resolve().parents[2] / "config.yaml"
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def check_health() -> dict:
    """返回两个组件的独立状态；不生成回答、加载Embedding或写入文档块。

    Ollama只读取本地模型列表，不能代替一次真实推理的验证。
    Chroma只访问已存在的集合；缺少索引时不自动建库来伪装可用。
    """
    config = load_config()
    llm = {"status": "error", "service_reachable": False, "model_available": None}
    started = perf_counter()
    try:
        settings = config["llm"]
        llm["model"] = settings["model"]
        url = urlparse(settings["base_url"])
        if settings["provider"] != "ollama" or url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("健康检查仅支持本地HTTP Ollama服务")
        # 不使用环境代理、不跟随重定向，避免检查请求离开配置的本地服务。
        with httpx.Client(timeout=3, trust_env=False, follow_redirects=False) as client:
            response = client.get(settings["base_url"].rstrip("/") + "/api/tags")
            response.raise_for_status()
            llm["service_reachable"] = True
            models = response.json()["models"]
        if not isinstance(models, list) or any(not isinstance(m, dict) or not isinstance(m.get("name"), str) for m in models):
            raise ValueError("Ollama模型列表格式无效")
        model = settings["model"] if ":" in settings["model"] else settings["model"] + ":latest"
        llm["model_available"] = any(m["name"] == model for m in models)
        llm.update(status="ok" if llm["model_available"] else "model_missing",
                   detail="本地服务可访问，配置模型已安装；未执行推理。" if llm["model_available"]
                   else "本地服务可访问，但配置模型未安装；请先准备对应模型。")
    except Exception as error:
        llm["detail"] = f"{type(error).__name__}: {error}。请检查Ollama启动状态、模型配置和服务地址后重试。"
    llm["seconds"] = perf_counter() - started

    database = {"status": "error", "chunks": None}
    started = perf_counter()
    try:
        retrieval = config["retrieval"]
        if retrieval["vector_store"] != "chroma":
            raise ValueError("当前项目只使用本地Chroma")
        database["collection"] = retrieval["collection_name"]
        directory = Path(__file__).resolve().parents[2] / config["paths"]["vector_index"]
        if directory.exists() and not directory.is_dir():
            raise ValueError("索引路径不是目录")
        if not (directory / "chroma.sqlite3").is_file():
            database.update(status="not_initialized", detail="尚未建立Chroma索引，请先导入文档；本次检查未自动建库。")
        else:
            # 延迟导入；沿用业务使用的真实Chroma客户端，禁用遥测和默认Embedding。
            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(path=str(directory.resolve()), settings=Settings(anonymized_telemetry=False))
            client.heartbeat()
            collection = client.get_collection(retrieval["collection_name"], embedding_function=None)
            expected = {"embedding_model": config["embedding"]["model"],
                        "embedding_revision": config["embedding"]["revision"], "hnsw:space": "cosine",
                        "hnsw:search_ef": retrieval["search_ef"], "hnsw:num_threads": retrieval["index_threads"]}
            if any((collection.metadata or {}).get(key) != value for key, value in expected.items()):
                raise ValueError("已有索引的模型版本或索引参数与配置不一致")
            database.update(status="ok", chunks=collection.count(), detail="Chroma心跳、集合配置及块数读取正常；未执行向量检索。")
    except Exception as error:
        database["detail"] = f"{type(error).__name__}: {error}。请检查索引目录、权限和集合配置后重试。"
    database["seconds"] = perf_counter() - started
    return {"status": "ok" if llm["status"] == database["status"] == "ok" else "degraded",
            "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "llm": llm, "vector_database": database}
