"""真实本地检索、Qwen 与 Streamlit AppTest 验证降级；不修改用户默认索引。"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from src.generation.prompt_template import PROMPT_VERSION
from src.generation.rag_pipeline import urlopen
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/5_2_3 缓存与降级策略/降级策略验证结果.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("请使用新的输出路径，保留原验证记录")
    config = load_config()
    dataset_path = PROJECT_ROOT / "reports/5_2_1 Prompt工程与生成策略/生成参数评测集.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    references = dataset["cases"][0]["context"]["references"]
    papers = dataset["papers"][:2]
    for paper in papers:
        path = PROJECT_ROOT / "data/raw/embedding_papers" / (paper["id"] + ".pdf")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == paper["pdf_sha256"]
    report = {"started_at": datetime.now().astimezone().isoformat(), "prompt_version": PROMPT_VERSION,
              "config": deepcopy(config), "environment": {"python": platform.python_version(), "platform": platform.platform()},
              "scope": "临时库中两篇公开论文的两个标注原文块，真实本地模型与AppTest；不是浏览器视觉验收或独立质量评测。",
              "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(), "papers": papers,
              "server": json.load(urlopen(config["llm"]["base_url"] + "/api/version", timeout=10)),
              "models": json.load(urlopen(config["llm"]["base_url"] + "/api/tags", timeout=10)), "requests": []}
    # 复用既有真实检索数据观察门槛，不重新标注或覆盖原实验。
    old_path = PROJECT_ROOT / "reports/5_1_4 混合检索与重排序/检索三档对比结果.json"
    old_queries = json.loads(old_path.read_text(encoding="utf-8"))["methods"]["hybrid_rerank"]["queries"]
    low_queries = [row for row in old_queries if row["top5"][0]["score"] < config["generation"]["low_relevance_threshold"]]
    report["threshold_observation"] = {"source_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
                                       "total": len(old_queries), "below_threshold": len(low_queries),
                                       "below_threshold_hit_at_5": sum(row["hit_at_5"] for row in low_queries),
                                       "below_threshold_top1_correct": sum(row["first_relevant_rank"] == 1 for row in low_queries),
                                       "note": "开发集观察，阈值不是概率或最优参数；高分仍可能不相关。"}
    searches = []
    real_search = HybridRetriever.search

    def counted_search(retriever, *positional, **kwargs):
        results = real_search(retriever, *positional, **kwargs)
        searches.append({"question": positional[0], "scores": [score for _, score in results]})
        return results

    with tempfile.TemporaryDirectory(prefix="rag-degradation-") as directory, ExitStack() as stack:
        config["paths"]["vector_index"] = str(Path(directory) / "index")
        config["paths"]["raw_documents"] = str(Path(directory) / "raw")
        config["paths"]["logs"] = str(Path(directory) / "logs")
        for target in ("src.utils.config", "src.retrieval.vector_store", "src.retrieval.hybrid_retriever",
                       "src.retrieval.reranker", "src.generation.rag_pipeline", "src.generation.cache", "src.utils.logger"):
            stack.enter_context(patch(target + ".load_config", return_value=config))
        stack.enter_context(patch.object(HybridRetriever, "search", counted_search))
        chat = stack.enter_context(patch("src.generation.streaming.urlopen", wraps=urlopen))
        store = VectorStore()
        app = AppTest.from_file(str(PROJECT_ROOT / "src/frontend/app.py"), default_timeout=60).run()

        def record(label, before, started):
            assert not app.exception
            message = deepcopy(app.session_state["rag_messages"][-1])
            report["requests"].append({"label": label, "elapsed_seconds": perf_counter() - started,
                                       "chat_requests": chat.call_count - before, "message": message,
                                       "visible_errors": [item.value for item in app.error],
                                       "visible_notices": [item.value for item in app.info]})
            return message

        started, before = perf_counter(), chat.call_count
        app.chat_input[0].set_value("请用一句话解释机器学习。").run()
        message = record("空库：纯模型概念回答", before, started)
        assert message["complete"] and message["generation_mode"] == "empty"
        assert message["answer"].startswith("当前知识库中未找到相关文档。") and not message["citations"]
        assert not app.session_state["rag_cache"].entries
        assert store.add_chunks([Document(page_content=ref["text"], metadata=ref["metadata"]) for ref in references]) == 2

        started, before = perf_counter(), chat.call_count
        app.chat_input[0].set_value("南京地铁的票价是多少？").run()
        assert not app.exception and "rag_pending" in app.session_state
        assert chat.call_count == before  # 确认前真实 HTTP 调用次数为零。
        pending = deepcopy(app.session_state["rag_pending"])
        report["pending"] = {"question": pending["message"]["question"], "context": pending["context"],
                             "chat_requests": chat.call_count - before, "search": searches[-1],
                             "visible_warnings": [item.value for item in app.warning],
                             "visible_candidates": [item.value for item in app.text]}
        app.button(key="confirm_low_relevance").click().run()
        message = record("低相关：确认后使用候选", before, started)
        assert message["complete"] and message["generation_mode"] == "low"
        assert "相关性低" in message["answer"] and not app.session_state["rag_cache"].entries

        # 实际关闭的本机端口与实际不存在的模型，分别触发连接错误和 HTTP 404。
        original_url, original_model = config["llm"]["base_url"], config["llm"]["model"]
        with socket.socket() as port:
            port.bind(("127.0.0.1", 0))
            unused_port = port.getsockname()[1]
        for label in ("连接失败", "模型不存在"):
            config["llm"]["base_url"] = f"http://127.0.0.1:{unused_port}" if label == "连接失败" else original_url
            config["llm"]["model"] = original_model if label == "连接失败" else "rag-degradation-missing-model-20260929"
            before, started = chat.call_count, perf_counter()
            app.chat_input[0].set_value("How many layers does BERT-base have?").run()
            message = record(label, before, started)
            assert not message["complete"] and message["type"] == "error"
            assert "重新提交问题" in message["retry_advice"] and not app.session_state["rag_cache"].entries
        config["llm"].update(base_url=original_url, model=original_model)
        report["searches"] = searches
    report["finished_at"] = datetime.now().astimezone().isoformat()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"真实降级验证通过：{args.output}")


if __name__ == "__main__":
    main()
