"""用真实本地模型和 AppTest 核对请求 JSONL、实际用量与检索分布。"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from src.generation.rag_pipeline import urlopen
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config
from src.utils.logger import read_rag_requests, request_time, retrieval_score_distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/5_2_4 日志与可观测性/RAG日志与检索分数验证结果.json")
    args = parser.parse_args()
    raw_output = args.output.with_suffix(".jsonl")
    if args.output.exists() or raw_output.exists():
        raise FileExistsError("请指定新的结果路径，保留已有实测记录")
    config = load_config()
    source = PROJECT_ROOT / "reports/5_2_1 Prompt工程与生成策略/生成参数评测集.json"
    dataset = json.loads(source.read_text(encoding="utf-8"))
    references = dataset["cases"][0]["context"]["references"]
    for paper in dataset["papers"][:2]:
        pdf = PROJECT_ROOT / "data/raw/embedding_papers" / (paper["id"] + ".pdf")
        assert hashlib.sha256(pdf.read_bytes()).hexdigest() == paper["pdf_sha256"]
    report = {"started_at": request_time(), "config": deepcopy(config),
              "environment": {"python": platform.python_version(), "platform": platform.platform()},
              "scope": "真实Qwen/M3E/BGE/临时Chroma与AppTest；两篇公开论文的两个标注块，不是独立质量或浏览器视觉评测。",
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "papers": dataset["papers"][:2],
              "server": json.load(urlopen(config["llm"]["base_url"] + "/api/version", timeout=10)),
              "models": json.load(urlopen(config["llm"]["base_url"] + "/api/tags", timeout=10)),
              "validation_complete": False, "requests": []}

    with tempfile.TemporaryDirectory(prefix="rag-log-verify-") as directory, ExitStack() as stack:
        for name, folder in (("raw_documents", "raw"), ("vector_index", "index"), ("logs", "logs")):
            config["paths"][name] = str(Path(directory) / folder)
        for target in ("src.utils.config", "src.utils.logger", "src.retrieval.vector_store",
                       "src.retrieval.hybrid_retriever", "src.retrieval.reranker",
                       "src.generation.rag_pipeline", "src.generation.cache"):
            stack.enter_context(patch(target + ".load_config", return_value=config))
        chat = stack.enter_context(patch("src.generation.streaming.urlopen", wraps=urlopen))
        store = VectorStore()
        app = AppTest.from_file(str(PROJECT_ROOT / "src/frontend/app.py"), default_timeout=60).run()

        def checkpoint():
            """逐次保留真实答案和文件快照，后续校验失败也不会丢失已取得的证据。"""
            raw = b"".join(path.read_bytes() for path in sorted(Path(config["paths"]["logs"]).glob("rag_*.jsonl")))
            report["distribution"] = retrieval_score_distribution()
            report["log_snapshots"] = len(raw.splitlines())
            report["log_sha256"] = hashlib.sha256(raw).hexdigest()
            raw_output.write_bytes(raw)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        def submit(label, question):
            before, started = chat.call_count, perf_counter()
            app.chat_input[0].set_value(question).run()
            assert not app.exception
            if "rag_pending" in app.session_state:
                assert chat.call_count == before
                app.button(key="cancel_low_relevance").click().run()
            records, invalid = read_rag_requests()
            assert not app.exception and invalid == 0
            record = records[-1]
            report["requests"].append({"label": label, "request_id": record["request_id"],
                                       "status": record["status"], "elapsed_seconds": perf_counter() - started,
                                       "generation_http_requests": chat.call_count - before,
                                       "tokens": record["tokens"], "top1_score": record["retrieval"]["top1_score"],
                                       "retrieved_chunks": len(record["retrieval"]["documents"]),
                                       "answer": record["answer"], "warnings": record["warnings"], "error": record["error"]})
            checkpoint()
            return record

        empty = submit("空库纯模型", "请用一句话解释机器学习。")
        assert empty["status"] == "completed" and empty["retrieval"]["status"] == "empty"
        assert empty["tokens"]["source"] == "ollama" and empty["tokens"]["total"] > 0
        assert empty["retrieval"]["top1_score"] is None
        assert store.add_chunks([Document(page_content=ref["text"], metadata=ref["metadata"]) for ref in references]) == 2

        question = "Answer in English: How many layers does BERT-base have? Cite [参考文档1] after the fact."
        normal = submit("正常论文问答", question)
        assert normal["status"] == "completed" and normal["tokens"]["total"] > 0
        assert len(normal["retrieval"]["documents"]) == 2
        assert normal["retrieval"]["documents"][0]["metadata"]["source_file"] == "bert.pdf"
        cacheable = bool(normal["citations"] and not normal["warnings"] and normal["done_reason"] == "stop")
        cached = submit("相同问题缓存复用检查", question)
        report["cache_probe"] = {"original_answer_cacheable": cacheable, "actual_hit": cached["cache"]["hit"]}
        assert cached["cache"]["hit"] == cacheable
        if cacheable:
            assert cached["tokens"]["total"] == 0
            assert cached["original_usage"]["eval_count"] == normal["tokens"]["output"]
            assert cached["retrieval"]["documents"] == [] and cached["retrieval"]["top1_score"] is None
        else:
            assert report["requests"][-1]["generation_http_requests"] == 1
            assert cached["retrieval"]["documents"]  # 不合格原答案不复用，真实检索计入分布。

        cancelled = submit("低相关候选取消", "南京地铁的票价是多少？")
        assert cancelled["status"] == "cancelled" and cancelled["tokens"]["total"] == 0
        assert cancelled["retrieval"]["top1_score"] < config["generation"]["low_relevance_threshold"]
        config["llm"]["model"] = "rag-log-missing-model-20260930"
        failure = submit("真实不存在的模型HTTP404", question)
        assert failure["status"] == "error" and "404" in failure["error"]
        assert failure["tokens"]["total"] is None and failure["tokens"]["source"] == "unavailable"

        report["distribution"] = retrieval_score_distribution()
        assert report["distribution"]["requests"] == 5 and report["distribution"]["cache_hits"] == int(cacheable)
        assert report["distribution"]["distributions"][0]["count"] == (3 if cacheable else 4)
        assert report["distribution"]["empty_retrievals"] == 1
        raw = b"".join(path.read_bytes() for path in sorted(Path(config["paths"]["logs"]).glob("rag_*.jsonl")))
        report["log_snapshots"] = len(raw.splitlines())
        report["log_sha256"] = hashlib.sha256(raw).hexdigest()
        report["visible_statistics"] = [item.value for item in app.caption if "请求 " in item.value or "样本 " in item.value]
        report["visible_chart_count"] = len(app.get("vega_lite_chart"))
        assert report["visible_chart_count"] == 1
    report["finished_at"] = request_time()
    report["validation_complete"] = True
    raw_output.write_bytes(raw)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"真实日志和分布验证通过：{args.output}")


if __name__ == "__main__":
    main()
