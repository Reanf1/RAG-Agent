"""补齐研究问题：Agent对固定RAG，以及真实独立任务串行/并行对照。"""
import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from io import BytesIO
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
from threading import Lock
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("baseline", Path(__file__).with_name("evaluate_system.py"))
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def supplemental():
    """通用概念/计算/时间题补充路由覆盖；不替代60题论文答案的人评。"""
    groups = [("concept", ["什么是深度学习？", "什么是机器学习？", "解释什么是人工智能。", "What is deep learning?"], []),
              ("calculator", ["3.14乘以2.56", "计算25加17", "计算144除以12", "计算9乘以8"], ["calculator"]),
              ("time", ["现在几点？", "返回当前系统时间和时区", "What is the current time?", "请查询今天的日期和当前时间"], ["current_time"])]
    return [{"id": f"X{i + 1:02}", "category": category, "language": "en" if text.isascii() else "zh",
             "question": text, "expected_tools": expected, "supplemental": True}
            for i, (category, text, expected) in enumerate((category, text, expected)
                for category, texts, expected in groups for text in texts)]


def aggregate(rows):
    """缺失实际用量保持未知，失败请求保留在响应和决策指标分母中。"""
    known = [r["tokens"]["total"] for r in rows if r["tokens"]["total"] is not None]
    return {"requests": len(rows), "seconds_mean": statistics.mean(r["seconds"] for r in rows),
            "tokens_mean_known": statistics.mean(known) if known else None,
            "unknown_requests": len(rows) - len(known), "tokens_total_known": sum(known),
            "iterations_mean": statistics.mean(r["iterations"] for r in rows),
            "tool_path_accuracy": statistics.mean(r["tool_selection_correct"] for r in rows),
            "by_category": {category: {"requests": len(part),
                 "tokens_mean_known": statistics.mean(r["tokens"]["total"] for r in part if r["tokens"]["total"] is not None),
                 "seconds_mean": statistics.mean(r["seconds"] for r in part),
                 "tool_path_accuracy": statistics.mean(r["tool_selection_correct"] for r in part)}
              for category in sorted({r["category"] for r in rows})
              for part in [[r for r in rows if r["category"] == category]]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("routing", "parallel"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, default=Path("/private/tmp/rag-evaluation-full-20261003"),
                        help="已完成evaluate_system检索阶段、含index/raw的冻结目录")
    args = parser.parse_args()
    if args.root.exists() or args.output.exists():
        raise FileExistsError("实验根目录和输出须为新路径")
    args.root.mkdir(parents=True)
    frozen = args.frozen_root
    for name in ("index", "raw"):
        shutil.copytree(frozen / name, args.root / name)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import torch
    torch.set_num_threads(4)
    import src.utils.config as config_module
    config = config_module.load_config()
    config["paths"].update(vector_index=str(args.root / "index"), raw_documents=str(args.root / "raw"),
                           logs=str(args.root / "logs"), session_db=str(args.root / "sessions.sqlite3"))
    config_module.load_config = lambda: deepcopy(config)
    from src.agent import react_loop
    from src.agent.tools import get_available_tools
    from src.agent.router import execute_calls
    from src.generation import rag_pipeline
    from src.agent.tools import knowledge_base_search
    from src.retrieval.hybrid_retriever import HybridRetriever
    # 评测资料固定为UTF-8，不能依赖Windows默认的GBK编码。
    questions = json.loads(baseline.DATASET.read_text(encoding="utf-8")) + supplemental()
    manifest = json.loads(baseline.MANIFEST.read_text(encoding="utf-8"))
    report = {"status": "running", "started_at": datetime.now().astimezone().isoformat(), "stage": args.stage,
              "config": deepcopy(config), "frozen_root": str(frozen), "model_seed": 20261004, "rows": [], "profiles": {}, "questions": questions,
              "inputs_sha256": {str(p.relative_to(ROOT)): baseline.digest(p) for p in [baseline.DATASET, baseline.MANIFEST, ROOT / "config.yaml"]},
              "source_sha256": {str(p.relative_to(ROOT)): baseline.digest(p) for p in sorted((ROOT / "src").rglob("*.py"))}}
    save = lambda: baseline.write_json(args.output, report)
    opener, calls, current, lock = rag_pipeline.urlopen, [], {}, Lock()
    raw_path = args.output.with_suffix(".calls.jsonl")
    def record(request, **kwargs):
        """仅记录实际HTTP响应并锁定种子，不替换模型、工具或语义结果。"""
        payload = json.loads(request.data) if getattr(request, "data", None) else None
        if payload:
            payload["options"]["seed"] = report["model_seed"]
            request.data = json.dumps(payload, ensure_ascii=False).encode()
        entry, start = {**current, "request": payload}, perf_counter()
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
    def usage(actual):
        unknown = sum(any(type(c.get("response", {}).get(key)) is not int for key in ("prompt_eval_count", "eval_count")) for c in actual)
        values = {key: sum(c.get("response", {}).get(key, 0) or 0 for c in actual) for key in ("prompt_eval_count", "eval_count")}
        return {**values, "total": None if unknown else sum(values.values()), "unknown_calls": unknown}
    tools = get_available_tools()
    # 预热不进入正式用量和延迟，保存真实服务版本/模型摘要。
    for endpoint in ("tags", "version"):
        with opener(config["llm"]["base_url"] + "/api/" + endpoint, timeout=10) as response:
            report[endpoint] = json.load(response)
    from langchain_core.messages import HumanMessage
    with opener(react_loop._model_request([HumanMessage(content="只回复好")]), timeout=300) as response:
        report["warmup"] = json.load(response)
    HybridRetriever().search("视觉Transformer", k=5, rerank=True)
    save()
    if args.stage == "routing":
        order = random.Random(20261004).sample(questions, len(questions))
        cases = [(q, profile, 1) for i, q in enumerate(order) for profile in
                 (("agent", "fixed_rag") if i % 2 == 0 else ("fixed_rag", "agent"))]
    else:
        ids = {p["id"]: p["doc_id"] for p in manifest["papers"]}
        tasks = [("time_keywords", "同时返回当前时间并用keyword_extract提取文本关键词：Transformer用于机器翻译，ViT用于图像分类。", ["current_time", "keyword_extract"]),
                 ("two_knowledge", f"用knowledge_base_search同时分别回答两项独立问题：论文ViT的图像分类输入是什么（doc_id={ids['vit']}）；论文DeiT的训练数据是什么（doc_id={ids['deit']}）。", ["knowledge_base_search"])]
        cases = [({"id": name, "category": name, "language": "zh", "question": text, "expected_tools": expected}, mode, repeat)
                 for repeat in (1, 2, 3) for name, text, expected in tasks
                 for mode in (("serial", "parallel") if repeat % 2 else ("parallel", "serial"))]
    for index, (q, profile, repeat) in enumerate(cases, 1):
        current.update(id=q["id"], profile=profile, repeat=repeat)
        begin, start = len(calls), perf_counter()
        error, events = None, []
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(rag_pipeline, "urlopen", record))
                stack.enter_context(patch.object(react_loop, "urlopen", record))
                if profile == "serial":
                    # 仅改变实际执行调度；模型规划、参数、工具结果及Observation仍走相同业务。
                    stack.enter_context(patch.object(react_loop, "execute_calls", lambda calls, tools, **kw: execute_calls(calls, tools, parallel=False)))
                if profile == "fixed_rag":
                    answer = knowledge_base_search.invoke({"question": q["question"]})
                    done = {"full_response": answer["answer"], "iterations": 0, "task_complete": answer["status"] == "answered", "context": {"observations": [{"result": answer}]}}
                    selections = [{"name": "knowledge_base_search", "args": {"question": q["question"]}}]
                else:
                    events = list(react_loop.run_react(q["question"], tools=tools, context={"history": []}))
                    done = events[-1]
                    selections = [{key: e[key] for key in ("name", "args", "iteration")} for e in events if e["type"] == "tool_call"]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            done, selections = {"full_response": error, "iterations": 0, "task_complete": False}, []
        elapsed, actual = perf_counter() - start, calls[begin:]
        correct = (set(c["name"] for c in selections) == set(q["expected_tools"])) if "expected_tools" in q else baseline.tool_selection(q, selections, manifest)
        report["rows"].append({"id": q["id"], "profile": profile, "repeat": repeat, "category": q["category"], "language": q["language"],
            "seconds": elapsed, "tokens": usage(actual), "iterations": done["iterations"], "task_complete": done["task_complete"],
            "tool_selection_correct": correct, "tool_calls": selections, "answer": done["full_response"], "error": error,
            "tool_results": [{k: e[k] for k in ("name", "status", "result", "iteration", "execution_mode", "elapsed_seconds")} for e in events if e["type"] == "tool_result"]})
        save()
        print(f"{index}/{len(cases)} {q['id']} {profile} {elapsed:.2f}s {usage(actual)['total']}Token complete={done['task_complete']}", flush=True)
    report["profiles"] = {p: aggregate([r for r in report["rows"] if r["profile"] == p]) for p in sorted({r["profile"] for r in report["rows"]})}
    report["tool_combinations"] = dict(Counter(" + ".join(sorted(set(c["name"] for c in r["tool_calls"]))) or "无工具" for r in report["rows"] if r["profile"] == "agent"))
    report.update(status="completed", completed_at=datetime.now().astimezone().isoformat())
    save()


if __name__ == "__main__":
    main()
