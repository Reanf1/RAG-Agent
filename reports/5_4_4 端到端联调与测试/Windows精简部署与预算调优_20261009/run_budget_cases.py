"""真实 Windows 预算对照：仅替换配置中的预算和副本路径，不替换检索、模型或工具。"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
from time import perf_counter
from unittest.mock import patch
from urllib.request import urlopen as service_open

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--case", choices=["compare", "fact", "summary", "multitool", "markdown"], required=True)
parser.add_argument("--num-predict", type=int, required=True)
parser.add_argument("--max-context", type=int, required=True)
parser.add_argument("--num-ctx", type=int, default=16384)
parser.add_argument("--name", required=True)
parser.add_argument("--confirm-candidates", action="store_true", help="测试审阅后复用低相关候选，不改变生产确认流程")
args = parser.parse_args()
sys.path.insert(0, str(args.project))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
modules = [importlib.import_module("src." + name) for name in (
    "agent.react_loop", "agent.tools", "agent.router", "agent.memory", "generation.rag_pipeline",
    "generation.prompt_template", "generation.cache", "retrieval.vector_store", "retrieval.hybrid_retriever",
    "retrieval.bm25_retriever", "retrieval.reranker", "utils.logger", "utils.token_budget")]
from src.utils.config import load_config
from src.agent.tools import get_available_tools
from src.agent.react_loop import run_react
from src.generation import rag_pipeline
from langchain_core.messages import message_to_dict

config = deepcopy(load_config())
config["llm"].update(num_ctx=args.num_ctx, num_predict=args.num_predict)
config["generation"].update(max_context_chars=args.max_context, max_prompt_chars=max(12000, args.max_context * 2))
for key, relative in (("raw_documents", "probe/raw"), ("vector_index", "probe/index"),
                      ("logs", "case-logs"), ("session_db", "case-memory.sqlite3")):
    config["paths"][key] = str(args.run / relative)
vit = "8ce7b83971a14508ca711a27c875c9b6914c4f6767cf3150fb1ca6c07aa056d6"
attention = "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697"
markdown = "6455abc0e87729a9bd5767f0fc835919c596456e7313a7fe0d1872ad6aa9c695"
output = args.run / (args.name + ".json")
if output.exists():
    raise FileExistsError("不覆盖已有预算结果")
calls_file = (args.run / (args.name + ".calls.jsonl")).open("x", encoding="utf-8")
original_open = rag_pipeline.urlopen
calls, resources, stop = [], [], Event()


class RecordedResponse:
    """原样转交真实响应；读完后保存终止原因、完整请求和实际用量。"""
    def __init__(self, response, payload, started):
        self.response, self.payload, self.started, self.parts = response, payload, started, []

    def __enter__(self):
        self.response.__enter__()
        return self

    def read(self, *arguments):
        data = self.response.read(*arguments)
        self.parts.append(data)
        return data

    def __iter__(self):
        for line in self.response:
            self.parts.append(line)
            yield line

    def __exit__(self, *arguments):
        self.response.__exit__(*arguments)
        raw = b"".join(self.parts).decode("utf-8")
        packets = [json.loads(line) for line in raw.splitlines() if line.strip()]
        last = packets[-1] if packets else {}
        row = dict(seconds=perf_counter() - self.started, request=self.payload, response=packets,
                   actual_input_tokens=last.get("prompt_eval_count"), actual_output_tokens=last.get("eval_count"),
                   done_reason=last.get("done_reason"), done=last.get("done"))
        calls.append({key: value for key, value in row.items() if key not in {"request", "response"}})
        calls_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        calls_file.flush()


def recording_open(request, *arguments, **keywords):
    started = perf_counter()
    payload = json.loads(request.data)
    payload["options"]["seed"] = 42  # 预算对照固定采样起点；真实请求／响应和实际用量仍完整记录。
    request.data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return RecordedResponse(original_open(request, *arguments, **keywords), payload, started)


def sample_resources():
    while not stop.is_set():
        row = {"at": datetime.now().astimezone().isoformat()}
        try:
            row["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"], text=True).strip()
            with service_open("http://127.0.0.1:11434/api/ps", timeout=3) as response:
                row["ollama"] = json.load(response)
        except Exception as error:
            row["error"] = repr(error)
        resources.append(row)
        stop.wait(1)


report = dict(started_at=datetime.now().astimezone().isoformat(), name=args.name, case=args.case,
              source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.project, text=True).strip(),
              config=config, calls=calls, resources=resources, boundary=__doc__)
sampler = Thread(target=sample_resources, daemon=True)
sampler.start()
started = perf_counter()
try:
    with ExitStack() as stack:
        for module in modules:
            if hasattr(module, "load_config"):
                stack.enter_context(patch.object(module, "load_config", return_value=config))
        stack.enter_context(patch.object(rag_pipeline, "urlopen", side_effect=recording_open))
        stack.enter_context(patch("src.agent.react_loop.urlopen", side_effect=recording_open))
        pending = {}
        tools = {tool.name: tool for tool in get_available_tools(session_id=args.name, pending=pending)}
        if args.case == "compare":
            arguments = dict(paper_a_id=attention, paper_b_id=vit)
            report["result"] = tools["paper_compare"].invoke(arguments)
            if args.confirm_candidates and report["result"].get("status") == "needs_confirmation":
                report["initial_result"] = report["result"]
                approval = pending[report["result"]["confirmation_id"]]
                confirmed = {tool.name: tool for tool in get_available_tools(session_id=args.name, confirmation=approval)}
                report["result"] = confirmed["paper_compare"].invoke(arguments)
                report["test_confirmation"] = "测试脚本按显式参数复用已记录候选；不代表用户质量审核通过。"
        elif args.case == "summary":
            report["result"] = tools["paper_summary"].invoke(dict(doc_id=vit))
        elif args.case == "fact":
            report["result"] = tools["knowledge_base_search"].invoke(dict(
                question="What pretraining datasets and image counts are used for ViT? Include ImageNet, ImageNet-21k and JFT. Answer in Chinese with page citations.", doc_id=vit))
        else:
            question = ("分别计算3.14*2.56和12.5/2，然后报告这两个计算结果。" if args.case == "multitool" else
                        "知识库中文档ID " + markdown + " 的Markdown标记是什么？注明行号。")
            report["question"] = question
            events = list(run_react(question, list(tools.values()), stream=True))
            report["events"] = [event for event in events if event["type"] != "token"]
            report["stream_tokens"] = sum(event["type"] == "token" for event in events)
            report["result"] = events[-1]
except Exception as error:
    report["error"] = repr(error)
finally:
    stop.set()
    sampler.join(timeout=5)
    calls_file.close()
    report.update(seconds=perf_counter()-started, finished_at=datetime.now().astimezone().isoformat())
    with output.open("x", encoding="utf-8") as stream:
        # Agent 保留 LangChain 消息；审计记录完整类型与内容，不能用 str 丢失结构。
        json.dump(report, stream, ensure_ascii=False, indent=2,
                  default=message_to_dict)
    result = report.get("result", {})
    print(json.dumps(dict(name=args.name, seconds=report["seconds"], error=report.get("error"),
                         calls=calls, status=result.get("status"), task_complete=result.get("task_complete"),
                         answer=result.get("answer", result.get("report")), warnings=result.get("warnings")), ensure_ascii=False), flush=True)
