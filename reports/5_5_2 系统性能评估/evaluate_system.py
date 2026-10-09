"""在锁定的60题上运行真实本地检索与Agent，保存逐题答案和实际模型用量。"""

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import sys
from threading import Lock
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATASET = ROOT / "reports/评测集.json"
MANIFEST = ROOT / "reports/5_5_1 评测集构建/论文清单.json"
CORPUS = ROOT / "data/raw/evaluation_vision_transformers/corpus.json"
PROFILES = {"vector": ("纯向量", 20, False), "hybrid": ("混合检索", 20, False),
            "rerank20": ("混合+重排/候选20", 20, True),
            "rerank10": ("混合+重排/候选10", 10, True),
            "rerank40": ("混合+重排/候选40", 40, True)}


def digest(path):
    """哈希用于冻结输入，不对运行后的答案反向调整标注。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def page_metrics(ranked, question):
    """按论文指纹与PDF物理页匹配；重复同页不会重复增加Recall。"""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paper_hashes = {p["id"]: p["doc_id"] for p in manifest["papers"]}
    gold = {(paper_hashes[e["paper_id"]], e["page_number"]) for e in question["evidence"]}
    found, first = set(), None
    for rank, item in enumerate(ranked[:5], 1):
        metadata = item["metadata"]
        start = metadata["page_number"]
        pages = {(metadata["doc_id"], page) for page in
                 range(start, metadata.get("page_end", start) + 1)}
        matches = pages & gold
        found |= matches
        if matches and first is None:
            first = rank
    gold_papers = {pid for pid, page in gold}
    found_papers = {pid for pid, page in found}
    return {"hit_at_5": bool(found), "mrr_at_5": 1 / first if first else 0,
            "recall_at_5": len(found) / len(gold), "first_relevant_rank": first,
            "gold_pages": len(gold), "matched_pages": len(found),
            "paper_coverage_at_5": len(found_papers) / len(gold_papers),
            "all_papers_hit_at_5": found_papers == gold_papers}


def summarize_retrieval(rows):
    keys = ("hit_at_5", "mrr_at_5", "recall_at_5", "paper_coverage_at_5", "all_papers_hit_at_5")
    return {"questions": len(rows), **{key: statistics.mean(r[key] for r in rows) for key in keys},
            "latency_mean_ms": statistics.mean(r["seconds"] * 1000 for r in rows),
            "by_category": {key: {metric: statistics.mean(r[metric] for r in rows if r["category"] == key)
                                  for metric in keys} for key in sorted({r["category"] for r in rows})},
            "by_language": {key: {metric: statistics.mean(r[metric] for r in rows if r["language"] == key)
                                  for metric in keys} for key in sorted({r["language"] for r in rows})}}


def tool_selection(question, calls, manifest):
    """预先锁定的可接受路径：RAG；对比题也可列表后使用真实论文对比。"""
    names = [call["name"] for call in calls]
    allowed = {"knowledge_base_search", "paper_list"}
    if question["category"] == "comparison":
        allowed.add("paper_compare")
    if not names or not set(names) <= allowed:
        return False
    if "knowledge_base_search" in names:
        return True
    expected = {p["doc_id"] for p in manifest["papers"] if p["id"] in question["paper_ids"]}
    compared = set()
    for call in calls:
        if call["name"] == "paper_compare":
            compared.update(call["args"].get(k) for k in ("paper_a_id", "paper_b_id"))
    return "paper_compare" in names and compared == expected


def summarize_agent(rows):
    """失败题仍进入准确率、轮次、耗时分母；缺失Token保持未知。"""
    totals = [r["tokens"]["total"] for r in rows if r["tokens"]["total"] is not None]
    return {"questions": len(rows), "tool_selection_accuracy": statistics.mean(r["tool_selection_correct"] for r in rows),
            "iterations_mean": statistics.mean(r["iterations"] for r in rows),
            "latency_mean_seconds": statistics.mean(r["seconds"] for r in rows),
            "task_complete_rate": statistics.mean(r["task_complete"] for r in rows),
            "tokens_total": sum(totals), "tokens_mean_known_requests": statistics.mean(totals) if totals else None,
            "token_complete_requests": len(totals), "token_unknown_requests": len(rows) - len(totals),
            "input_tokens_known": sum(r["tokens"]["input_known"] for r in rows),
            "output_tokens_known": sum(r["tokens"]["output_known"] for r in rows),
            "stop_reasons": dict(Counter(r["stop_reason"] for r in rows))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("retrieval", "agent"), required=True)
    parser.add_argument("--root", type=Path, required=True, help="本轮隔离工作目录；检索阶段须不存在")
    parser.add_argument("--output", type=Path, required=True, help="本阶段新结果JSON路径")
    parser.add_argument("--limit", type=int, default=60, help="只供先行验证；正式运行为60题")
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.limit <= 60:
        raise ValueError("结果必须用新路径，limit范围1～60")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import torch
    torch.set_num_threads(4)
    from langchain_core.documents import Document
    import src.utils.config as config_module
    config = config_module.load_config()
    original_config = deepcopy(config)
    config["paths"].update(vector_index=str(args.root / "index"), raw_documents=str(args.root / "raw"),
                           logs=str(args.root / "logs"), session_db=str(args.root / "sessions.sqlite3"))
    # 只覆盖评测进程中读取的路径，业务YAML和已有索引不变；调用仍执行真实模块。
    config_module.load_config = lambda: deepcopy(config)
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.agent import react_loop
    from src.agent.tools import get_available_tools
    from src.generation import rag_pipeline
    questions = json.loads(DATASET.read_text(encoding="utf-8"))[:args.limit]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    assert corpus["manifest_sha256"] == digest(MANIFEST)
    # 片段数量由冻结加载器实际生成，不能把历史1694块当作当前版本的固定数量。
    assert corpus["corpus"] and len({row["id"] for row in corpus["corpus"]}) == len(corpus["corpus"])
    assert {row["paper_id"] for row in corpus["corpus"]} == {paper["id"] for paper in manifest["papers"]}
    fingerprint = {"dataset_sha256": digest(DATASET), "manifest_sha256": digest(MANIFEST),
                   "corpus_sha256": digest(CORPUS), "config_sha256": digest(ROOT / "config.yaml")}
    fingerprint["evaluation_script_sha256"] = digest(Path(__file__))
    fingerprint["source_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src").rglob("*.py"))}
    report = {"started_at": datetime.now().astimezone().isoformat(), "status": "running",
              "stage": args.stage, "environment": {"platform": platform.platform(), "python": platform.python_version(),
                    "torch_threads": torch.get_num_threads()}, "inputs": fingerprint,
              "business_config": original_config, "experiment_config": deepcopy(config),
              "question_count": len(questions), "profiles": {}, "rows": [], "model_seed": 20261003,
              "human_scoring": "pending_independent_review", "annotation": "论文+物理页；非完整chunk金标准；ViT来源与开发集重叠"}
    if args.stage == "retrieval":
        args.root.mkdir(parents=True, exist_ok=False)
        for paper in manifest["papers"]:
            original = ROOT / paper["local_path"]
            assert digest(original) == paper["doc_id"]
            directory = args.root / "raw" / paper["doc_id"]
            directory.mkdir(parents=True)
            shutil.copy2(original, directory / original.name)
        documents = [Document(page_content=row["text"], metadata=row["metadata"]) for row in corpus["corpus"]]
        store = VectorStore()
        started = perf_counter()
        assert store.add_chunks(documents) == len(documents)
        report["build_seconds"] = perf_counter() - started
        write_json(args.root / "input_fingerprint.json", fingerprint)
        print(f"索引准备完成：{store.count()}块，{report['build_seconds']:.2f}秒", flush=True)
        retriever = HybridRetriever(store)
        def search(profile, text):
            label, count, rerank = PROFILES[profile]
            retriever.candidate_k = count
            return store.search(text, k=5) if profile == "vector" else retriever.search(text, k=5, rerank=rerank)
        started = perf_counter()
        for profile in PROFILES:
            search(profile, questions[0]["question"])
        report["warmup_seconds"] = perf_counter() - started
        write_json(args.output, report)
        profiles = list(PROFILES)
        order = random.Random(20261003).sample(questions, len(questions))
        for index, question in enumerate(order):
            offset = index % len(profiles)
            for profile in profiles[offset:] + profiles[:offset]:
                started = perf_counter()
                found = search(profile, question["question"])
                elapsed = perf_counter() - started
                ranked = [{"metadata": deepcopy(doc.metadata), "score": score, "text": doc.page_content}
                          for doc, score in found]
                report["rows"].append({"id": question["id"], "profile": profile, "category": question["category"],
                       "language": question["language"], "seconds": elapsed, "top5": ranked,
                       **page_metrics(ranked, question)})
                write_json(args.output, report)
            print(f"检索 {index+1}/{len(questions)}：{question['id']} 五组完成", flush=True)
        report["profiles"] = {key: {"label": value[0], "candidate_k": value[1], "rerank": value[2],
                **summarize_retrieval([r for r in report["rows"] if r["profile"] == key])} for key, value in PROFILES.items()}
    else:
        assert json.loads((args.root / "input_fingerprint.json").read_text(encoding="utf-8")) == fingerprint
        store = VectorStore()
        assert store.count() == len(corpus["corpus"])
        with rag_pipeline.urlopen(config["llm"]["base_url"] + "/api/tags", timeout=10) as response:
            report["model_tags"] = json.load(response)
        with rag_pipeline.urlopen(config["llm"]["base_url"] + "/api/version", timeout=10) as response:
            report["server"] = json.load(response)
        # 预热单列，不进入60题的平均延迟和Token统计。
        from urllib.request import Request
        request = Request(config["llm"]["base_url"] + "/api/chat", data=json.dumps({"model": config["llm"]["model"],
                    "stream": False, "messages": [{"role": "user", "content": "只回复：好"}],
                    "options": {"num_predict": 8, "num_ctx": config["llm"]["num_ctx"]}}).encode(),
                    headers={"Content-Type": "application/json"})
        started = perf_counter()
        with rag_pipeline.urlopen(request, timeout=300) as response:
            report["warmup_response"] = json.load(response)
        report["warmup_seconds"] = perf_counter() - started
        started = perf_counter()
        HybridRetriever().search("视觉Transformer的训练方法", k=5, rerank=True)
        report["retrieval_warmup_seconds"] = perf_counter() - started
        # 真正的工具/模型仍按现有业务执行，patch只记录本地非流式HTTP响应和做路由消融。
        opener, lock, current, calls = rag_pipeline.urlopen, Lock(), {}, []
        raw_path = args.output.with_suffix(".calls.jsonl")
        if raw_path.exists():
            raise FileExistsError(raw_path)
        def record(request, **kwargs):
            payload = json.loads(request.data) if getattr(request, "data", None) else None
            if payload is not None:
                # 两组使用相同实验种子，只影响本轮本地采样，不改业务默认配置。
                payload["options"]["seed"] = report["model_seed"]
                request.data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            start, entry = perf_counter(), {**current, "request": payload}
            try:
                with opener(request, **kwargs) as response:
                    data = response.read()
                entry["response"] = json.loads(data)
                return BytesIO(data)
            except Exception as error:
                entry["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                entry["seconds"] = perf_counter() - start
                with lock:
                    calls.append(entry)
                    with raw_path.open("a", encoding="utf-8") as output:
                        output.write(json.dumps(entry, ensure_ascii=False) + "\n")
        tools = get_available_tools()
        write_json(args.output, report)
        for index, question in enumerate(questions):
            profiles = ["default", "no_rules"] if index % 2 == 0 else ["no_rules", "default"]
            for profile in profiles:
                current.update(id=question["id"], profile=profile)
                before = len(calls)
                with ExitStack() as stack:
                    stack.enter_context(patch.object(rag_pipeline, "urlopen", record))
                    stack.enter_context(patch.object(react_loop, "urlopen", record))
                    if profile == "no_rules":
                        stack.enter_context(patch.object(react_loop, "route_question", lambda *a, **k: None))
                    started = perf_counter()
                    events = list(react_loop.run_react(question["question"], tools=tools, context={"history": []}))
                    elapsed = perf_counter() - started
                done = events[-1]
                assert done["type"] == "done"
                actual = calls[before:]
                unknown = sum(any(type(c.get("response", {}).get(k)) is not int for k in
                                  ("prompt_eval_count", "eval_count")) for c in actual)
                inputs = sum(c.get("response", {}).get("prompt_eval_count", 0) or 0 for c in actual)
                outputs = sum(c.get("response", {}).get("eval_count", 0) or 0 for c in actual)
                selections = [{k: e[k] for k in ("name", "args", "iteration")} for e in events if e["type"] == "tool_call"]
                report["rows"].append({"id": question["id"], "profile": profile, "category": question["category"],
                    "language": question["language"], "answer": done["full_response"], "task_complete": done["task_complete"],
                    "stop_reason": done["stop_reason"], "iterations": done["iterations"], "seconds": elapsed,
                    "model_calls": len(actual), "tokens": {"input_known": inputs, "output_known": outputs,
                        "total": None if unknown else inputs + outputs, "unknown_calls": unknown, "source": "actual_ollama_responses"},
                    "tool_calls": selections, "tool_selection_correct": tool_selection(question, selections, manifest),
                    "event_types": [e["type"] for e in events], "metrics": done["metrics"]})
                write_json(args.output, report)
                print(f"Agent {index+1}/{len(questions)} {question['id']} {profile}：{elapsed:.2f}秒，"
                      f"{done['iterations']}轮，{inputs+outputs}已知Token，{done['stop_reason']}", flush=True)
        report["profiles"] = {key: {"label": label, "rule_routing": key == "default",
                 **summarize_agent([r for r in report["rows"] if r["profile"] == key])}
                              for key, label in (("default", "默认规则路由+ReAct"), ("no_rules", "关闭规则路由的ReAct"))}
    report.update(status="completed", completed_at=datetime.now().astimezone().isoformat())
    write_json(args.output, report)
    print(json.dumps(report["profiles"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
