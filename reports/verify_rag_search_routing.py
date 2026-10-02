"""模块四来源决策实测：真实本地模型/论文/索引，外网可用性单独记录。"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from reports.verify_error_recovery import serial
from src.agent.react_loop import run_react, think
from src.agent.tools import get_available_tools
from src.data_loader import create_import_tasks
from src.retrieval.vector_store import VectorStore, batch_build_index
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError("请使用新路径，保留已有结果")
    config = deepcopy(load_config())
    config["agent"]["online_search_enabled"] = False
    report = {"started_at": datetime.now().astimezone().isoformat(), "rows": [],
              "scope": "仅隔离配置路径和临时切换已有联网开关；模型、上传、索引和搜索实际执行。"
                       "单篇公开AI论文和功能问题不构成独立路由准确率评测；搜索服务可用性另记。"}

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def agent_case(name, question, check):
        row = {"name": name, "question": question, "events": []}
        report["rows"].append(row)
        for event in run_react(question):
            row["events"].append(serial(event))
            save()
            print(name, event["type"], event.get("name", ""), flush=True)
        row["passed"] = bool(check(row["events"][-1], row["events"]))
        save()
        return row

    def planned_case(name, question, expected, context=None):
        row = {"name": name, "question": question, "context": context, "kind": "real_model_planning"}
        report["rows"].append(row)
        try:
            row["thought"] = think(question, get_available_tools(), context)
            row["passed"] = row["thought"]["tool_name"] == expected
        except Exception as error:
            row.update(passed=False, error=f"{type(error).__name__}: {error}")
        save()
        print(name, row["passed"], flush=True)

    with tempfile.TemporaryDirectory(prefix="rag-search-routing-") as directory, ExitStack() as stack:
        for key, folder in (("raw_documents", "raw"), ("vector_index", "index"), ("logs", "logs")):
            config["paths"][key] = str(Path(directory) / folder)
        for module in ("src.agent.tools", "src.agent.react_loop", "src.agent.router",
                       "src.retrieval.vector_store", "src.retrieval.hybrid_retriever", "src.retrieval.reranker",
                       "src.generation.rag_pipeline", "src.utils.logger", "src.chunking"):
            stack.enter_context(patch(module + ".load_config", return_value=config))
        tasks = create_import_tasks([("attention.pdf", (ROOT / "data/raw/embedding_papers/attention.pdf").read_bytes())])
        for progress in batch_build_index(tasks, config["paths"]["raw_documents"], vector_store=VectorStore()):
            if progress["completed"]:
                print("入库", tasks[0]["status"], tasks[0].get("chunk_count"), flush=True)
        if tasks[0]["status"] != "success" or not tasks[0].get("indexed"):
            raise RuntimeError("真实论文入库失败")
        doc_id = tasks[0]["documents"][0].metadata["doc_id"]
        report["ingestion"] = {"doc_id": doc_id, "file": "attention.pdf", "chunks": tasks[0]["chunk_count"]}
        save()
        agent_case("local_rag_citations", "知识库中Transformer在WMT2014英德翻译任务上的BLEU是多少？请在答案中标注提供的参考文档编号。",
                   lambda done, events: done["task_complete"] and "28.4" in done["full_response"] and any(
                       e["type"] == "tool_result" and e["name"] == "knowledge_base_search" and e["status"] == "success"
                       and e["result"].get("citations") and all(c["metadata"]["doc_id"] == doc_id
                           for c in e["result"]["citations"]) for e in events))
        agent_case("offline_latest_unavailable", "最新Transformer论文有哪些？",
                   lambda done, events: not done["task_complete"] and done["stop_reason"] == "incomplete"
                   and not any(e["type"] == "tool_call" for e in events) and "联网" in done["full_response"])
        config["agent"]["online_search_enabled"] = True
        planned_case("model_local_scope_with_history", "这篇已上传论文报告的最新BLEU结果是多少？文献ID：" + doc_id,
                     "knowledge_base_search", {"history": [{"role": "human", "content": "本轮只核对已上传论文原文。"}]})
        planned_case("model_external_freshness", "最近发布的Transformer研究论文有哪些？", "web_search")
        row = agent_case("live_external_search", "联网搜索最新Transformer论文",
                         lambda done, events: any(e["type"] == "tool_result" and e["name"] == "web_search" for e in events)
                         and not any(e["type"] == "tool_call" and e["name"] == "knowledge_base_search" for e in events)
                         and (done["task_complete"] if any(e["type"] == "tool_result" and e["status"] == "success"
                              and e["result"].get("results") for e in events) else not done["task_complete"]))
        search_results = [e for e in row["events"] if e["type"] == "tool_result" and e["name"] == "web_search"]
        report["search_service_available"] = any(e["status"] == "success" and e["result"].get("results") for e in search_results)
        config["paths"]["vector_index"] = str(Path(directory) / "empty-index")
        agent_case("empty_local_no_auto_web", "知识库中关于这篇论文的实验数据集是什么？",
                   lambda done, events: "当前知识库中未找到相关文档" in done["full_response"] and any(
                       e["type"] == "tool_result" and e["name"] == "knowledge_base_search" and e["status"] == "success"
                       and e["result"]["generation_mode"] == "empty" for e in events)
                   and not any(e["type"] == "tool_call" and e["name"] == "web_search" for e in events))
    report["finished_at"] = datetime.now().astimezone().isoformat()
    report["functional_checks_passed"] = all(row["passed"] for row in report["rows"])
    save()
    if not report["functional_checks_passed"]:
        raise RuntimeError("存在未通过的功能检查，真实轨迹已保留")


if __name__ == "__main__":
    main()
