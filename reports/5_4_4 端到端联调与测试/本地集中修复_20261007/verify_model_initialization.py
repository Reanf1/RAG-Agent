"""新进程以三个实际线程验证M3E／BGE首次加载单例；不是性能或Windows验收。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import sys
from threading import Barrier
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from sentence_transformers import CrossEncoder
import src.retrieval.vector_store as vectors
import src.retrieval.reranker as reranking
from src.utils.config import load_config


def main():
    output = Path(__file__).with_name("真实模型并发初始化.json")
    if output.exists():
        raise FileExistsError("不覆盖历史结果，请使用新的报告目录")
    config = load_config()
    report = {"scope": __doc__, "started_at": datetime.now().astimezone().isoformat(),
              "embedding": config["embedding"], "reranker": config["retrieval"]["reranker_local_path"],
              "results": [], "passed": False}
    try:
        for name, getter, target, constructor in (
                ("M3E", vectors.get_embeddings, "src.retrieval.vector_store.HuggingFaceEmbeddings", vectors.HuggingFaceEmbeddings),
                ("BGE", reranking.get_reranker, "sentence_transformers.CrossEncoder", CrossEncoder)):
            getter.cache_clear()
            barrier = Barrier(3)
            def load(_):
                barrier.wait(timeout=10)
                return getter()
            # 包装器只记录实际构造次数，真实权重构造及加载照常执行。
            with patch(target, wraps=constructor) as counted, ThreadPoolExecutor(3) as pool:
                models = list(pool.map(load, range(3)))
            row = {"model": name, "threads": 3, "constructions": counted.call_count,
                   "same_instance": all(model is models[0] for model in models)}
            row["passed"] = row["constructions"] == 1 and row["same_instance"]
            report["results"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        report["passed"] = all(row["passed"] for row in report["results"])
    finally:
        report["finished_at"] = datetime.now().astimezone().isoformat()
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise AssertionError("真实模型并发初始化未通过，已有结果已保留")


if __name__ == "__main__":
    main()
