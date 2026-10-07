"""Windows真实权重和RAG缓存复测；只使用独立索引，耗时不作为性能基准。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
from threading import Barrier
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.root.exists() or args.output.exists():
        raise FileExistsError("不覆盖已有测试数据或证据")
    sys.path.insert(0, str(args.project))
    os.environ.setdefault("HF_HOME", str(args.project / "data/models/.hf-runtime"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from sentence_transformers import CrossEncoder
    import src.retrieval.vector_store as vectors
    import src.retrieval.reranker as reranking
    from src.generation.cache import SemanticCache, cache_scope
    from src.generation.rag_pipeline import generate_answer
    from src.agent.tools import _knowledge_base_search
    from src.utils.config import load_config

    config = deepcopy(load_config())
    args.root.mkdir(parents=True)
    shutil.copytree(args.source / "raw", args.root / "raw")
    shutil.copytree(args.source / "index", args.root / "index")
    for key, name in (("raw_documents", "raw"), ("vector_index", "index"), ("logs", "logs")):
        config["paths"][key] = str(args.root / name)
    report = {"scope": __doc__, "started_at": datetime.now().astimezone().isoformat(),
              "model_initialization": [], "cache_pairs": [], "rag": {}, "passed": False}
    try:
        for name, getter, target, constructor in (
                ("M3E", vectors.get_embeddings, "src.retrieval.vector_store.HuggingFaceEmbeddings", vectors.HuggingFaceEmbeddings),
                ("BGE", reranking.get_reranker, "sentence_transformers.CrossEncoder", CrossEncoder)):
            getter.cache_clear()
            barrier = Barrier(3)
            def load(_):
                barrier.wait(timeout=10)
                return getter()
            # 只计数真实构造次数，权重及初始化逻辑不替换。
            with patch(target, wraps=constructor) as counted, ThreadPoolExecutor(3) as pool:
                models = list(pool.map(load, range(3)))
            row = {"model": name, "threads": 3, "constructions": counted.call_count,
                   "same_instance": all(model is models[0] for model in models)}
            report["model_initialization"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            assert row["constructions"] == 1 and row["same_instance"]
        # 复用已有九对条件样例；此部分的固定缓存答案不作论文生成质量证据。
        examples = runpy.run_path(str(args.project / "reports/5_4_4 端到端联调与测试/本地集中修复_20261007/verify_cache_constraints.py"))
        protocol = {"type": "done", "answer": "固定缓存协议样例。", "done_reason": "stop",
                    "generation_mode": "grounded", "citations": [{"id": 1}], "usage": {"eval_count": 0}}
        for expected, left, right in examples["PAIRS"]:
            cache = SemanticCache()
            assert cache.put(left, protocol, "pair-proof")
            hit = cache.lookup(right, "pair-proof")
            row = {"cached_question": left, "query": right, "expected_hit": expected,
                   "actual_hit": hit is not None, "cache": hit.get("cache") if hit else None}
            report["cache_pairs"].append(row)
            assert row["actual_hit"] == expected
        with ExitStack() as stack:
            for module in ("src.agent.tools", "src.retrieval.vector_store", "src.retrieval.hybrid_retriever",
                           "src.retrieval.reranker", "src.generation.rag_pipeline", "src.generation.cache", "src.utils.logger"):
                stack.enter_context(patch(module + ".load_config", return_value=config))
            cache = SemanticCache()
            question = "TXT文件中的复测代号是什么？"
            doc_id = "cb990ff1e44de00ba328e34169099d3f4704470d7d34fa9b71a586471d715baa"
            generation = stack.enter_context(patch("src.generation.rag_pipeline.generate_answer", wraps=generate_answer))
            first = _knowledge_base_search(question, doc_id, cache=cache, session_id="windows-cache-proof")
            report["rag"]["first"] = first
            assert first["status"] == "answered" and first["citations"] and generation.call_count == 1
            exact = _knowledge_base_search(question, doc_id, cache=cache, session_id="windows-cache-proof")
            report["rag"]["exact"] = exact
            assert exact["cache"]["mode"] == "exact" and generation.call_count == 1
            assert all(value == 0 for value in exact["usage"].values())
            semantic = _knowledge_base_search("TXT文件的复测代号是什么？", doc_id, cache=cache,
                                               session_id="windows-cache-proof")
            report["rag"]["semantic"] = semantic
            assert semantic["cache"]["mode"] == "semantic" and generation.call_count == 1
            store = vectors.VectorStore()
            scope = cache_scope(store) + ":" + doc_id
            assert cache.lookup(question, scope + "-other-document") is None
            assert cache.put(question, {"type": "done", **first}, scope)
            config["llm"]["temperature"] += .01
            assert cache.lookup(question, cache_scope(store) + ":" + doc_id) is None
            config["llm"]["temperature"] -= .01
            assert cache.put(question, {"type": "done", **first}, cache_scope(store) + ":" + doc_id)
            chunk = store.list_chunks(doc_id)[0]
            before = store.count()
            # 同数量正文替换只发生于本脚本的索引副本，随后不再对改写正文进行检索。
            vector = store._store.get(ids=[chunk.metadata["chunk_id"]], include=["embeddings"])["embeddings"]
            store._store._collection.update(ids=[chunk.metadata["chunk_id"]],
                                            documents=[chunk.page_content + "\n测试变更"], embeddings=vector.tolist())
            assert store.count() == before
            assert cache.lookup(question, cache_scope(store) + ":" + doc_id) is None
            report["rag"].update(generation_calls=generation.call_count, document_scope_invalidated=True,
                                 config_invalidated=True, same_count_corpus_invalidated=True)
        report["passed"] = True
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        report["finished_at"] = datetime.now().astimezone().isoformat()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("PASSED=" + str(report["passed"]), flush=True)


if __name__ == "__main__":
    main()
