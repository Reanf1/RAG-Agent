"""用完整AI论文、真实本地模型和临时索引验证两个科研工具及Agent注册调用。"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import BaseMessage

from src.agent.react_loop import run_react
from src.agent.tools import AVAILABLE_TOOLS, execute_tool
from src.data_loader import create_import_tasks
from src.retrieval.vector_store import VectorStore, batch_build_index
from src.utils.config import load_config
from src.utils.logger import request_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/科研工具验证结果_20261002.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("请指定不存在的新路径，保留历史验证证据")
    config = deepcopy(load_config())
    report = {"started_at": request_time(), "validation_complete": False,
              "scope": "完整Attention/ViT公开PDF及明确标注的中文功能样例；真实本地模型/上传/索引/工具/Agent，"
                       "只替换临时存储目录，不模拟返回，不计为系统答案质量或元信息准确率评测。",
              "rows": []}

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="rag-research-tools-") as directory, ExitStack() as stack:
        config["paths"]["raw_documents"] = str(Path(directory) / "raw")
        config["paths"]["vector_index"] = str(Path(directory) / "index")
        config["paths"]["logs"] = str(Path(directory) / "logs")
        for module in ("src.agent.tools", "src.agent.react_loop", "src.retrieval.vector_store",
                       "src.retrieval.hybrid_retriever", "src.retrieval.reranker", "src.generation.rag_pipeline", "src.utils.logger"):
            stack.enter_context(patch(module + ".load_config", return_value=config))
        report["config"] = config
        files = [(name + ".pdf", (ROOT / "data/raw/embedding_papers" / (name + ".pdf")).read_bytes())
                 for name in ("attention", "vit")]
        chinese = ("中文人工智能模型评测\n作者：张三、李四\n发表年份：2024\n摘要\n"
                   "本文比较人工智能模型的准确率与推理速度。\n实验显示不同模型适用于不同任务。\n"
                   "关键词：人工智能\nDOI: 10.1234/demo.2024\n")
        files.append(("中文功能样例.md", chinese.encode()))
        tasks = create_import_tasks(files)
        store = VectorStore()
        for progress in batch_build_index(tasks, config["paths"]["raw_documents"], vector_store=store):
            if progress["completed"]:
                print("入库", progress, flush=True)
        assert all(task.get("indexed") and task["status"] == "success" for task in tasks)
        report["ingestion"] = [{"file": task["name"], "doc_id": task["documents"][0].metadata["doc_id"],
                                "chunks": task["chunk_count"]} for task in tasks]
        save()
        ids = [task["documents"][0].metadata["doc_id"] for task in tasks]
        for doc_id, title, author, year in zip(ids, ("Attention Is All You Need", "AN IMAGE IS WORTH", "中文人工智能模型评测"),
                                              ("Ashish Vaswani", "Alexey Dosovitskiy", "张三"), (2017, 2021, 2024)):
            event = execute_tool("paper_metadata", {"doc_id": doc_id}, AVAILABLE_TOOLS)
            row = {"name": "metadata", "doc_id": doc_id,
                   "event": {key: value.model_dump() if isinstance(value, BaseMessage) else value for key, value in event.items()}}
            result = event.get("result") or {}
            row["passed"] = (event["status"] == "success" and title.lower() in (result.get("title") or "").lower()
                             and author in result.get("authors", []) and result.get("year") == year
                             and len(result.get("authors", [])) == (2 if doc_id == ids[-1] else 8 if doc_id == ids[0] else 12)
                             and bool(result.get("abstract")) and bool(result.get("evidence")))
            if doc_id == ids[-1]:
                row["passed"] &= result.get("doi") == "10.1234/demo.2024"
            else:
                row["passed"] &= result.get("doi") is None
            report["rows"].append(row)
            save()
            print("元信息", row["passed"], result.get("title"), result.get("year"), event["error"], flush=True)
        question = "Attention论文中Transformer在WMT 2014英德翻译任务的BLEU分数是多少？请在回答正文使用[参考文档N]格式引用，不能合并多个编号。"
        event = execute_tool("knowledge_base_search", {"question": question, "doc_id": ids[0]}, AVAILABLE_TOOLS)
        result = event.get("result") or {}
        row = {"name": "rag", "event": {key: value.model_dump() if isinstance(value, BaseMessage) else value for key, value in event.items()},
               "passed": event["status"] == "success" and result.get("status") == "answered"
                         and "28.4" in result.get("answer", "") and bool(result.get("citations"))}
        report["rows"].append(row)
        save()
        print("RAG", row["passed"], result.get("answer"), event["error"], flush=True)
        row = {"name": "agent_metadata", "events": []}
        report["rows"].append(row)
        for event in run_react("请使用paper_metadata提取attention.pdf的标题、作者和年份，依据真实工具结果回答。",
                               context={"papers": [{"doc_id": ids[0], "source_file": "attention.pdf"}]}):
            row["events"].append({key: value.model_dump() if isinstance(value, BaseMessage) else value for key, value in event.items()})
            save()
            print("Agent", event["type"], event.get("stop_reason", ""), flush=True)
        done = row["events"][-1]
        row["passed"] = (done["task_complete"] and done["stop_reason"] == "task_complete"
                         and bool(done["context"]["observations"])
                         and done["context"]["observations"][0]["name"] == "paper_metadata"
                         and "2017" in done["full_response"])
        from src.utils.logger import read_rag_requests
        report["rag_logs"], report["invalid_log_lines"] = read_rag_requests()
        save()
    report["validation_complete"], report["finished_at"] = True, request_time()
    save()
    assert all(row["passed"] for row in report["rows"]), "存在实测失败，保留原始结果供分析"


if __name__ == "__main__":
    main()
