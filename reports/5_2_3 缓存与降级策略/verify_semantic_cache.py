"""真实本地模型与 Streamlit AppTest 验证语义缓存，计时和调用次数均取实际执行。"""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import platform
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from src.generation.cache import _constraints, cache_scope
from src.generation.rag_pipeline import urlopen
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore, get_embeddings
from src.utils.config import load_config


PAIRS = [
    (True, "原始Transformer用了多少个注意力头？", "原始Transformer使用了几个注意力头？"),
    (True, "BERT-base的层数、隐藏维度和注意力头数分别是多少？", "BERT-base有几层、隐藏层维度多大、用了几个注意力头？"),
    (True, "How many layers does BERT-base have?", "How many layers are in BERT-base?"),
    (True, "What is the hidden size of BERT-base?", "What is BERT-base's hidden size?"),
    (False, "BERT-base的层数是多少？", "BERT-large的层数是多少？"),
    (False, "BERT-base的层数是多少？", "BERT-base的隐藏维度是多少？"),
    (False, "Transformer用了多少个注意力头？", "Transformer没有使用多头注意力吗？"),
    (False, "BERT使用15%的掩码比例吗？", "BERT使用20%的掩码比例吗？"),
    (False, "How many layers does BERT-base have?", "How many layers does BERT-large have?"),
    (False, "How many layers does BERT-base have?", "How many attention heads does BERT-base have?"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/5_2_3 缓存与降级策略/语义缓存验证结果.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("请指定新结果路径，保留已有实验记录")
    config = load_config()
    dataset_path = PROJECT_ROOT / "reports/5_2_1 Prompt工程与生成策略/生成参数评测集.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    references = dataset["cases"][0]["context"]["references"]
    papers = dataset["papers"][:2]
    for paper in papers:
        path = PROJECT_ROOT / "data/raw/embedding_papers" / (paper["id"] + ".pdf")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == paper["pdf_sha256"]
    report = {"started_at": datetime.now().astimezone().isoformat(), "config": deepcopy(config),
              "scope": "两篇真实论文的两个标注片段，实际检索/生成与AppTest；小型开发验证，不是独立质量评测或浏览器视觉检查。",
              "environment": {"python": platform.python_version(), "platform": platform.platform()},
              "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(), "papers": papers,
              "server": json.load(urlopen(config["llm"]["base_url"] + "/api/version", timeout=10)),
              "models": json.load(urlopen(config["llm"]["base_url"] + "/api/tags", timeout=10)),
              "validation_complete": False, "pairs": [], "requests": []}
    embedding = get_embeddings()
    for equivalent, left, right in PAIRS:
        a, b = embedding.embed_query(left), embedding.embed_query(right)
        cosine = sum(x * y for x, y in zip(a, b)) / (sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5
        compatible = _constraints(left) == _constraints(right)
        raw_hit = cosine >= config["generation"]["cache"]["similarity_threshold"]
        report["pairs"].append({"equivalent_by_author": equivalent, "cached_question": left, "query": right,
                                "cosine": cosine, "constraints_match": compatible,
                                "threshold_only_hit": raw_hit, "effective_hit": raw_hit and compatible})
    # 包装真实方法只计次数，原始算法与HTTP请求照常执行；不提供假模型返回。
    calls = []
    real_search = HybridRetriever.search

    def counted_search(retriever, *positional, **kwargs):
        calls.append(positional[0])
        return real_search(retriever, *positional, **kwargs)

    with tempfile.TemporaryDirectory(prefix="rag-cache-verify-") as directory:
        config["paths"]["raw_documents"] = str(Path(directory) / "raw")
        config["paths"]["vector_index"] = str(Path(directory) / "index")
        config["paths"]["logs"] = str(Path(directory) / "logs")
        config_targets = ("src.utils.config", "src.retrieval.vector_store", "src.retrieval.hybrid_retriever",
                          "src.retrieval.reranker", "src.generation.rag_pipeline", "src.generation.cache", "src.utils.logger")
        from contextlib import ExitStack
        with ExitStack() as stack:
            for target in config_targets:
                stack.enter_context(patch(target + ".load_config", return_value=config))
            stack.enter_context(patch.object(HybridRetriever, "search", counted_search))
            chat = stack.enter_context(patch("src.generation.streaming.urlopen", wraps=urlopen))
            store = VectorStore()
            docs = [Document(page_content=ref["text"], metadata=ref["metadata"]) for ref in references]
            assert store.add_chunks(docs) == 2
            app = AppTest.from_file(str(PROJECT_ROOT / "src/frontend/app.py"), default_timeout=60).run()
            questions = [
                "原始Transformer用了多少个注意力头？请在结论后标注[参考文档1]。",
                "原始Transformer用了多少个注意力头？请在结论后标注[参考文档1]。",
                "原始Transformer使用了几个注意力头？请在结论后标注[参考文档1]。",
                "Answer in English: How many layers does BERT-base have? Cite [参考文档1] after the fact.",
                "Answer in English: How many layers does BERT-base have? Cite [参考文档1] after the fact.",
                "Answer in English: How many layers are in BERT-base? Cite [参考文档1] after the fact.",
            ]
            for index, question in enumerate(questions):
                before = (len(calls), chat.call_count)
                started = perf_counter()
                app.chat_input[0].set_value(question).run()
                wall = perf_counter() - started
                assert not app.exception
                message = deepcopy(app.session_state["rag_messages"][-1])
                row = {"index": index, "message": message, "app_run_seconds": wall,
                       "retrieval_calls": len(calls) - before[0], "llm_calls": chat.call_count - before[1]}
                report["requests"].append(row)
                # 先保存真实答案：引用质量不合格时，断言失败也保留原因与已有探测数据。
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                assert message["complete"] and not message.get("error") and not message["warnings"], message.get("warnings")
                if index % 3:
                    assert message["cache"]["mode"] == ("exact" if index % 3 == 1 else "semantic")
                    assert (row["retrieval_calls"], row["llm_calls"]) == (0, 0)
                    first = report["requests"][index - index % 3]["message"]
                    assert message["answer"] == first["answer"] and message["citations"] == first["citations"]
                    assert message["usage"]["eval_count"] == 0
                else:
                    assert (row["retrieval_calls"], row["llm_calls"]) == (1, 1)
                print(index, message["elapsed_seconds"], message.get("cache", {"hit": False}), flush=True)
            old_scope = cache_scope(store)
            deleted = store.delete_document(docs[0].metadata["doc_id"])
            assert deleted == 1 and cache_scope(store) != old_scope
            assert app.session_state["rag_cache"].lookup(questions[0], cache_scope(store)) is None
            report["invalidation"] = {"deleted_chunks": deleted, "lookup_after_delete": None}
            report["rendered_cache_notices"] = [item.value for item in app.info]
            report["rendered_metrics"] = [item.value for item in app.caption if "Token" in item.value]
    report["loaded_models"] = json.load(urlopen(config["llm"]["base_url"] + "/api/ps", timeout=10))
    report["finished_at"] = datetime.now().astimezone().isoformat()
    report["validation_complete"] = True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
