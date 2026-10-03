"""核验真实Ollama和Chroma健康状态；异常路径隔离，用户索引不写入。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import chromadb
from chromadb.config import Settings

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from src.retrieval.vector_store import VectorStore
from src.utils.config import check_health, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError("请使用新输出文件，保留原核验结果")
    report = {"scope": "真实本地Ollama模型列表、现有Chroma读操作；未执行生成、Embedding或向量查询。异常路径使用临时目录。", "rows": []}
    config = load_config()
    # 现有用户索引只用get_collection访问，核验准备也不能自动创建集合。
    directory = (ROOT / config["paths"]["vector_index"]).resolve()
    if not (directory / "chroma.sqlite3").is_file():
        raise FileNotFoundError("请先准备现有Chroma索引，核验脚本不自动建用户数据库")
    client = chromadb.PersistentClient(path=str(directory),
                                       settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(config["retrieval"]["collection_name"], embedding_function=None)
    before = collection.get(include=[])["ids"]
    metadata = deepcopy(collection.metadata)

    def check(name, settings, llm_status, db_status):
        # 仅替换核验进程内配置；HTTP和ChromaAPI实际执行，不替换返回结果。
        with patch("src.utils.config.load_config", return_value=settings):
            result = check_health()
        report["rows"].append({"name": name, "result": result,
            "passed": result["llm"]["status"] == llm_status and result["vector_database"]["status"] == db_status})

    check("configured_service_and_index", config, "ok", "ok")
    changed = deepcopy(config)
    changed["llm"]["model"] = "health-check-not-installed:latest"
    check("model_missing", changed, "model_missing", "ok")
    changed = deepcopy(config)
    changed["llm"]["base_url"] = "http://127.0.0.1:11435"
    check("service_unavailable", changed, "error", "ok")
    with tempfile.TemporaryDirectory(prefix="rag-health-") as directory:
        changed = deepcopy(config)
        missing = Path(directory) / "missing"
        changed["paths"]["vector_index"] = str(missing)
        check("index_not_initialized", changed, "ok", "not_initialized")
        report["missing_directory_preserved"] = not missing.exists()
        empty = Path(directory) / "empty"
        VectorStore(empty)  # 测试准备：显式创建空库，健康检查本身不初始化。
        changed["paths"]["vector_index"] = str(empty)
        check("existing_empty_index", changed, "ok", "ok")
        corrupt = Path(directory) / "corrupt"
        corrupt.mkdir()
        (corrupt / "chroma.sqlite3").write_bytes(b"corrupt sqlite")
        changed["paths"]["vector_index"] = str(corrupt)
        check("corrupt_index", changed, "ok", "error")
    report["index_unchanged"] = collection.get(include=[])["ids"] == before and collection.metadata == metadata
    report["existing_chunks"] = len(before)
    report["passed"] = report["index_unchanged"] and report["missing_directory_preserved"] and all(row["passed"] for row in report["rows"])
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "chunks": len(before), "cases": len(report["rows"])}, ensure_ascii=False))
    if not report["passed"]:
        raise RuntimeError("有健康检查未通过，实际返回已保留")


if __name__ == "__main__":
    main()
