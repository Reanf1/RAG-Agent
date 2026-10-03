"""指标功能核验：真实本地模型、SQLite和公开论文，临时索引不改用户数据。"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from io import BytesIO
import json
from importlib import import_module
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

serial = import_module("reports.5_3_4 多轮对话记忆管理.verify_session_isolation").serial
from src.agent import MemoryManager, run_session
from src.agent import react_loop
from src.data_loader import create_import_tasks
from src.generation import rag_pipeline
from src.retrieval.vector_store import VectorStore, batch_build_index
from src.utils.config import load_config
from src.utils.logger import read_rag_requests, retrieval_request_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError("请换用新输出路径，保留已有核验")
    report = {"started_at": datetime.now().astimezone().isoformat(), "rows": [],
              "scope": "真实Ollama/M3E/BGE/Chroma，单篇公开论文的指标功能核验，候选命中率不是Hit@5。"}
    config = deepcopy(load_config())
    real_urlopen = react_loop.urlopen

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="agent-metrics-") as directory, ExitStack() as stack:
        for key, folder in (("raw_documents", "raw"), ("vector_index", "index"), ("logs", "logs")):
            config["paths"][key] = str(Path(directory) / folder)
        memory = MemoryManager(Path(directory) / "memory.sqlite3")
        for module in ("src.agent.tools", "src.agent.react_loop", "src.agent.router", "src.agent.memory",
                       "src.retrieval.vector_store", "src.retrieval.hybrid_retriever", "src.retrieval.reranker",
                       "src.generation.rag_pipeline", "src.utils.logger", "src.chunking"):
            stack.enter_context(patch(module + ".load_config", return_value=config))

        def check_case(name, question, *, unknown=False):
            row = {"name": name, "question": question, "events": [], "model_responses": []}
            report["rows"].append(row)

            def capture(request, *args, **kwargs):
                with real_urlopen(request, *args, **kwargs) as response:
                    data = response.read()
                row["model_responses"].append({"request": json.loads(request.data), "response": json.loads(data)})
                save()
                return BytesIO(data)  # 原始响应字节保持不变，只记录实际usage。

            with patch.object(react_loop, "urlopen", side_effect=capture), \
                    patch.object(rag_pipeline, "urlopen", side_effect=capture):
                session = memory.create_session("metrics-verifier")
                for event in run_session(question, "metrics-verifier", session, memory=memory):
                    row["events"].append(serial(event))
                    print(name, event["type"], event["metrics"]["tokens"], flush=True)
                    save()
            final = row["events"][-1]
            actual_total = sum(packet["response"][key] for packet in row["model_responses"]
                               for key in ("prompt_eval_count", "eval_count"))
            row["actual_response_tokens"] = actual_total
            row["token_check"] = final["metrics"]["tokens"]["total"] is None if unknown \
                else final["metrics"]["tokens"]["total"] == actual_total
            row["live_snapshot_check"] = len({e["metrics"]["tokens"]["known_total"] for e in row["events"]}) > 1 if not unknown else True
            row["rag_statistics"] = retrieval_request_metrics()
            save()
            return row

        check_case("calculator_actual_usage", "请用计算器计算3.14乘以2.56。")
        tasks = create_import_tasks([("attention.pdf", (ROOT / "data/raw/embedding_papers/attention.pdf").read_bytes())])
        for _ in batch_build_index(tasks, config["paths"]["raw_documents"], vector_store=VectorStore()):
            pass
        if tasks[0]["status"] != "success" or not tasks[0].get("indexed"):
            raise RuntimeError("公开论文实际入库失败")
        report["ingestion"] = {"file": "attention.pdf", "chunks": tasks[0]["chunk_count"]}
        row = check_case("rag_actual_usage", "知识库中Transformer在WMT2014英德翻译任务上的BLEU是多少？请引用参考文档编号。")
        row["rag_check"] = any(e["type"] == "tool_result" and e["name"] == "knowledge_base_search"
            and e["result"].get("retrieval", {}).get("returned_chunks", 0) > 0 for e in row["events"])
        config["generation"]["low_relevance_threshold"] = 0.999999  # 控制低相关性分支，仍执行真实检索和重排。
        row = check_case("low_relevance_no_generation_tokens", "知识库中的Transformer实验结果是什么？")
        row["rag_check"] = any(e["type"] == "tool_result" and e["name"] == "knowledge_base_search"
            and e["result"].get("status") == "needs_confirmation"
            and e["result"]["usage"] == {"prompt_eval_count": 0, "eval_count": 0} for e in row["events"])
        config["paths"]["vector_index"] = str(Path(directory) / "empty-index")
        row = check_case("empty_retrieval", "知识库中的论文使用什么实验数据集？")
        row["rag_check"] = row["rag_statistics"]["completed"] == 3 and row["rag_statistics"]["hits"] == 2
        report["rag_records"] = read_rag_requests()[0]
        config["llm"]["base_url"] = "http://127.0.0.1:11435"
        check_case("model_unavailable_unknown_usage", "请用计算器计算3.14乘以2.56。", unknown=True)
        report["agent_records"] = [json.loads(line) for path in Path(config["paths"]["logs"]).glob("agent_*.jsonl")
                                   for line in path.read_text().splitlines()]
    report["passed"] = all(r["token_check"] and r["live_snapshot_check"] and r.get("rag_check", True) for r in report["rows"])
    report["finished_at"] = datetime.now().astimezone().isoformat()
    save()
    if not report["passed"]:
        raise RuntimeError("核验未全部通过，实际响应及事件已保留")


if __name__ == "__main__":
    main()
