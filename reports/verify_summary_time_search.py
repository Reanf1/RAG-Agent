"""真实论文、本地模型、系统时钟和可选DuckDuckGo功能核验；不替代独立质量评测。"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timedelta
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from docx import Document
from langchain_core.messages import BaseMessage

from src.agent.react_loop import run_react
from src.agent.tools import AVAILABLE_TOOLS, execute_tool, get_available_tools, web_search
from src.data_loader import batch_import, create_import_tasks
from src.utils.config import load_config


def serial(value):
    """仅将消息转为JSON；真实结果、原文和错误完整保留。"""
    if isinstance(value, BaseMessage):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serial(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("输出已存在，请换一个路径，保留原始实验")
    config = deepcopy(load_config())
    report = {"started_at": datetime.now().astimezone().isoformat(), "config": config,
              "scope": "两篇完整公开AI论文与一个中文DOCX功能样例，不是独立质量准确率实验",
              "rows": [], "validation_complete": False}

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        config["paths"]["raw_documents"] = str(Path(directory) / "raw")
        config["paths"]["logs"] = str(Path(directory) / "logs")
        for module in ("src.agent.tools", "src.agent.react_loop", "src.generation.rag_pipeline", "src.chunking"):
            stack.enter_context(patch(module + ".load_config", return_value=config))
        # 只隔离配置路径；原文解析、分块、模型、HTTP、时钟均使用真实实现。
        word, buffer = Document(), BytesIO()
        for paragraph in ("农业课程问答的检索增强研究（人工构造的中文功能样例，并非真实发表论文）",
                          "背景：通用语言模型缺少农业课程专业资料，回答容易出错。",
                          "方法：使用本地模型，将文档切成512字符块，融合向量和BM25检索，再进行重排序。",
                          "结果：本样例设定50道课程题，其中43道回答正确，正确率86%；这是构造数据，不是项目实测。",
                          "结论：在本样例范围内，课程资料检索能够辅助回答；仍需核对来源，不能推广为所有农业问题的效果。"):
            word.add_paragraph(paragraph)
        word.save(buffer)
        inputs = [(name, (ROOT / "data/raw/embedding_papers" / name).read_bytes()) for name in ("attention.pdf", "vit.pdf")]
        inputs.append(("中文功能样例.docx", buffer.getvalue()))
        tasks = create_import_tasks(inputs)
        list(batch_import(tasks, config["paths"]["raw_documents"]))
        import hashlib
        for task in tasks:
            if task["status"] != "success":
                raise RuntimeError(task["error"])
            doc_id = hashlib.sha256(task["data"]).hexdigest()
            event = execute_tool("paper_summary", {"doc_id": doc_id}, AVAILABLE_TOOLS)
            result = event.get("result") or {}
            passed = (event["status"] == "success" and result.get("status") == "answered"
                      and len(result.get("sections", {})) == 4 and bool(result.get("citations"))
                      and all(ref["metadata"]["doc_id"] == doc_id for ref in result.get("citations", [])))
            report["rows"].append({"name": task["name"], "event": serial(event), "passed": passed})
            save()
            print(task["name"], passed, result.get("answer", event["error"]), flush=True)
        before = datetime.now().astimezone() - timedelta(seconds=1)
        event = execute_tool("current_time", {}, AVAILABLE_TOOLS)
        now = datetime.fromisoformat(event["result"]["system_time"])
        report["rows"].append({"name": "current_time", "event": serial(event),
                               "passed": before <= now <= datetime.now().astimezone() and now.utcoffset() is not None})
        event = execute_tool("web_search", {"query": "Attention Is All You Need paper"}, [web_search])
        report["rows"].append({"name": "disabled_search", "event": serial(event),
                               "passed": event["status"] == "error" and web_search not in get_available_tools()})
        save()
        row = {"name": "agent_time", "events": []}
        report["rows"].append(row)
        for event in run_react("请调用current_time，告诉我当前系统时间和时区，不要推测。"):
            row["events"].append(serial(event))
            save()
            print("Agent", event["type"], event.get("stop_reason", ""), flush=True)
        done = row["events"][-1]
        row["passed"] = (done["task_complete"] and done["stop_reason"] == "task_complete"
                         and any(item["name"] == "current_time" and item["status"] == "success"
                                 for item in done["context"].get("observations", [])))
        config["agent"]["online_search_enabled"] = True
        event = execute_tool("web_search", {"query": "Attention Is All You Need paper"}, get_available_tools())
        report["rows"].append({"name": "live_search", "event": serial(event), "optional_network": True,
                               "passed": event["status"] == "success" and bool((event.get("result") or {}).get("results"))})
        print("DuckDuckGo", event["status"], event.get("result"), event["error"], flush=True)
    report["validation_complete"], report["finished_at"] = True, datetime.now().astimezone().isoformat()
    save()
    assert all(row["passed"] for row in report["rows"] if not row.get("optional_network")), "本地功能存在实测失败，已保存原始证据"


if __name__ == "__main__":
    main()
