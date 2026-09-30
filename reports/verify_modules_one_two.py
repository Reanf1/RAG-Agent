"""完整 PDF 入库后用真实本地模型测试模块一、二的连接，保留质量问题。"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from src.data_loader import create_import_tasks
from src.generation.prompt_template import PROMPT_VERSION
from src.generation.rag_pipeline import urlopen
from src.retrieval.vector_store import VectorStore, batch_build_index
from src.utils.config import load_config
from src.utils.logger import read_rag_requests, request_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/模块一二连接验证结果_20260930.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("请使用新的结果路径，保留已有实测证据")
    config = load_config()
    dataset_path = ROOT / "reports/生成参数评测集.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    files, papers = [], []
    for paper in dataset["papers"][:2]:
        path = ROOT / "data/raw/embedding_papers" / (paper["id"] + ".pdf")
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == paper["pdf_sha256"]
        files.append((path.name, raw))
        papers.append(paper)
    report = {"started_at": request_time(), "prompt_version": PROMPT_VERSION, "config": deepcopy(config),
              "papers": papers, "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
              "scope": "两篇完整公开PDF真实入库和中英问答开发探测；数字出现不是答案准确率，引用定位不是语义正确性。",
              "validation_complete": False, "requests": []}
    with tempfile.TemporaryDirectory(prefix="rag-modules-one-two-") as directory, ExitStack() as stack:
        for key, folder in (("raw_documents", "raw"), ("vector_index", "index"), ("logs", "logs")):
            config["paths"][key] = str(Path(directory) / folder)
        # 仅替换隔离目录；实际加载、切分、编码、检索、重排与HTTP调用全部保留。
        for target in ("src.utils.config", "src.utils.logger", "src.retrieval.vector_store",
                       "src.retrieval.hybrid_retriever", "src.retrieval.reranker",
                       "src.generation.rag_pipeline", "src.generation.cache"):
            stack.enter_context(patch(target + ".load_config", return_value=config))
        store = VectorStore()
        tasks = create_import_tasks(files)
        progress = list(batch_build_index(tasks, config["paths"]["raw_documents"], vector_store=store))
        assert all(task["indexed"] and task["status"] == "success" for task in tasks)
        assert progress[-1] == {"completed": 2, "total": 2}
        report["ingestion"] = {"chunks": store.count(), "progress": progress,
                               "documents": [{key: task[key] for key in ("name", "chunk_count", "added_chunks", "status")}
                                             for task in tasks]}
        source_chunks = {doc.metadata["chunk_id"]: doc for doc in store.list_chunks()}
        app = AppTest.from_file(str(ROOT / "src/frontend/app.py"), default_timeout=120).run()
        assert not app.exception
        for case in dataset["cases"][:2]:
            app.chat_input[0].set_value(case["question"]).run()
            assert not app.exception
            pending = "rag_pending" in app.session_state
            if pending:
                app.button(key="confirm_low_relevance").click().run()
                assert not app.exception
            message = deepcopy(app.session_state["rag_messages"][-1])
            records, invalid = read_rag_requests()
            assert invalid == 0
            record = records[-1]
            # 数字规则仅用于发现遗漏；保留回答与原文供审阅，不作为通过的条件。
            body = re.split(r"^##\s+(?:参考来源|Reference Sources|References|Sources)\s*$",
                            message.get("raw_answer", ""), maxsplit=1, flags=re.M | re.I)[0]
            row = {"case_id": case["id"], "expected_note": case["expected_note"], "required_confirmation": pending,
                   "message": message, "log": record,
                   "fact_pattern_hits": [bool(re.search(pattern, body, re.I | re.A)) for pattern in case["fact_patterns"]]}
            report["requests"].append(row)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            assert message["complete"] and not message.get("error") and record["status"] == "completed"
            assert len(record["retrieval"]["documents"]) == config["retrieval"]["top_k"]
            assert message["usage"]["prompt_eval_count"] > 0 and message["usage"]["eval_count"] > 0
            for ref in message["context"]["references"] + message["citations"]:
                doc = source_chunks[ref["metadata"]["chunk_id"]]
                assert ref["metadata"] == doc.metadata and doc.page_content.startswith(ref["text"])
            assert record["answer"] == message["answer"]
            print(case["id"], "功能通过；数字规则", row["fact_pattern_hits"], "引用警告", message["warnings"], flush=True)
    report["loaded_models"] = json.load(urlopen(config["llm"]["base_url"] + "/api/ps", timeout=10))
    report["finished_at"], report["validation_complete"] = request_time(), True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
