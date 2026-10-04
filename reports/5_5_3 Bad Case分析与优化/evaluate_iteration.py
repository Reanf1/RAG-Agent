"""一轮Bad Case优化后完整60题真实复测；基线代码与原始结果保持不变。"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from threading import Lock
from time import perf_counter
from unittest.mock import patch

import importlib.util

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# 指标沿用已锁定的5.5.2定义，避免优化后改变评分口径。
spec = importlib.util.spec_from_file_location("baseline_evaluator", ROOT / "reports/5_5_2 系统性能评估/evaluate_system.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
DATASET, MANIFEST, CORPUS = baseline.DATASET, baseline.MANIFEST, baseline.CORPUS
digest, write_json = baseline.digest, baseline.write_json
tool_selection, summarize_agent = baseline.tool_selection, baseline.summarize_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.limit = 60
    if args.root.exists() or args.output.exists():
        raise FileExistsError("复测须使用全新的隔离目录与结果路径")
    lock = json.loads(Path(__file__).with_name("迭代方案_锁定.json").read_text())
    for name, expected in lock["inputs_sha256"].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in lock["baseline_sha256"].items():
        assert digest(ROOT / "reports/5_5_2 系统性能评估" / name) == expected, name
    args.root.mkdir(parents=True)
    # 复制冻结索引及同一批原文，不改变基线工作目录或已有用户知识库。
    source = Path("/private/tmp/rag-evaluation-full-20261003")
    for name in ("index", "raw"):
        shutil.copytree(source / name, args.root / name)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import torch
    torch.set_num_threads(4)
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
    assert len(corpus["corpus"]) == 1694
    fingerprint = {"dataset_sha256": digest(DATASET), "manifest_sha256": digest(MANIFEST),
                   "corpus_sha256": digest(CORPUS), "config_sha256": digest(ROOT / "config.yaml")}
    fingerprint["evaluation_script_sha256"] = digest(Path(__file__))
    fingerprint["source_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src").rglob("*.py"))}
    report = {"started_at": datetime.now().astimezone().isoformat(), "status": "running",
              "stage": "agent", "environment": {"platform": platform.platform(), "python": platform.python_version(),
                    "torch_threads": torch.get_num_threads()}, "inputs": fingerprint,
              "business_config": original_config, "experiment_config": deepcopy(config),
              "question_count": len(questions), "profiles": {}, "rows": [], "model_seed": 20261003,
              "human_scoring": "pending_independent_review", "annotation": "论文+物理页；非完整chunk金标准；ViT来源与开发集重叠"}
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
    # 真正的工具/模型仍按现有业务执行，patch只记录本地非流式HTTP响应；不改路由或工具实现。
    opener, lock, current, calls = rag_pipeline.urlopen, Lock(), {}, []
    raw_path = args.output.with_suffix(".calls.jsonl")
    if raw_path.exists():
        raise FileExistsError(raw_path)
    def record(request, **kwargs):
        payload = json.loads(request.data) if getattr(request, "data", None) else None
        if payload is not None:
            # 沿用基线的实验种子，只影响本轮本地采样，不改业务默认配置。
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
        profiles = ["default"]
        for profile in profiles:
            current.update(id=question["id"], profile=profile)
            before = len(calls)
            with ExitStack() as stack:
                stack.enter_context(patch.object(rag_pipeline, "urlopen", record))
                stack.enter_context(patch.object(react_loop, "urlopen", record))
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
                          for key, label in (("default", "一轮优化后的默认规则路由+ReAct"),)}
    report.update(status="completed", completed_at=datetime.now().astimezone().isoformat())
    write_json(args.output, report)
    print(json.dumps(report["profiles"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
