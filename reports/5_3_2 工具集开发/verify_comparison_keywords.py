"""完整AI论文、本地模型和临时索引的对比/关键词功能验证，不作为独立质量评测。"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("请使用新的报告路径，不覆盖历史记录")
    report = {"started_at": request_time(), "validation_complete": False, "rows": [],
              "scope": "完整Attention/ViT PDF，真实上传/索引/检索/本地模型及默认Agent；只替换临时存储路径。"}

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def serial(event):
        return {key: value.model_dump() if isinstance(value, BaseMessage) else value for key, value in event.items()}

    config = deepcopy(load_config())
    with tempfile.TemporaryDirectory(prefix="rag-compare-keywords-") as directory, ExitStack() as stack:
        for key, name in (("raw_documents", "raw"), ("vector_index", "index"), ("logs", "logs")):
            config["paths"][key] = str(Path(directory) / name)
        for module in ("src.agent.tools", "src.agent.react_loop", "src.retrieval.vector_store",
                       "src.retrieval.hybrid_retriever", "src.retrieval.reranker", "src.generation.rag_pipeline", "src.utils.logger"):
            stack.enter_context(patch(module + ".load_config", return_value=config))
        report["config"] = config
        tasks = create_import_tasks([(name + ".pdf", (ROOT / "data/raw/embedding_papers" / (name + ".pdf")).read_bytes())
                                     for name in ("attention", "vit")])
        for event in batch_build_index(tasks, config["paths"]["raw_documents"], vector_store=VectorStore()):
            if event["completed"]:
                print("入库", event, flush=True)
        assert all(task["status"] == "success" and task["indexed"] for task in tasks)
        ids = [task["documents"][0].metadata["doc_id"] for task in tasks]
        report["ingestion"] = [{"file": task["name"], "doc_id": identifier, "chunks": task["chunk_count"]}
                               for task, identifier in zip(tasks, ids)]
        save()
        cases = [("comparison", "paper_compare", {"paper_a_id": ids[0], "paper_b_id": ids[1]}),
                 ("question_keywords", "keyword_extract", {"text": "比较Transformer与ViT在机器翻译和图像分类任务上的数据集和准确率。"}),
                 ("document_keywords", "keyword_extract", {"doc_id": ids[0]})]
        for name, tool_name, arguments in cases:
            event = execute_tool(tool_name, arguments, AVAILABLE_TOOLS)
            result = event.get("result") or {}
            passed = event["status"] == "success"
            if name == "comparison":
                cited = {ref["metadata"]["doc_id"] for ref in result.get("citations", [])}
                answer = result.get("answer", "")
                passed &= (result.get("status") == "answered" and cited == set(ids)
                           and "Transformer" in answer and ("ViT" in answer or "Vision Transformer" in answer)
                           and "WMT" in answer and ("ImageNet" in answer or "JFT" in answer)
                           and result.get("done_reason") == "stop")
            else:
                passed &= bool(result.get("keywords")) and len(result["keywords"]) <= 5 and all(
                    ref["locations"] for ref in result.get("evidence", []))
            report["rows"].append({"name": name, "event": serial(event), "passed": passed})
            save()
            print(name, passed, result.get("answer", result.get("keywords")), event["error"], flush=True)
        row = {"name": "agent_keywords", "events": []}
        report["rows"].append(row)
        for event in run_react("请调用keyword_extract，提取这句话中的关键词：Transformer用于机器翻译，ViT用于图像分类。"):
            row["events"].append(serial(event))
            save()
            print("Agent", event["type"], event.get("stop_reason", ""), flush=True)
        done = row["events"][-1]
        row["passed"] = (done["task_complete"] and done["stop_reason"] == "task_complete"
                         and bool(done["context"]["observations"])
                         and done["context"]["observations"][0]["name"] == "keyword_extract")
    report["validation_complete"], report["finished_at"] = True, request_time()
    save()
    assert all(row["passed"] for row in report["rows"]), "存在实测失败，保留原始结果供分析"


if __name__ == "__main__":
    main()
