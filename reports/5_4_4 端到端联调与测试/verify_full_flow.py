"""核验真实页面完整流程留下的原文、Chroma、SQLite和请求日志，不替换模型结果。"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent.memory import MemoryManager
from src.chunking import split_documents
from src.data_loader.pdf_loader import load_pdf
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config
from src.utils.logger import request_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="隔离页面的测试数据目录")
    parser.add_argument("--output", type=Path, required=True, help="新的证据文件，禁止覆盖历史结果")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("请使用新的证据路径，保留首次失败与各次复测")
    config = deepcopy(load_config())
    config["paths"]["raw_documents"] = str(args.root / "raw")
    config["paths"]["vector_index"] = str(args.root / "index")
    config["paths"]["logs"] = str(args.root / "logs")
    config["paths"]["session_db"] = str(args.root / "memory.sqlite3")
    report = {"checked_at": request_time(), "root": str(args.root), "config": config,
              "scope": "真实页面测试后的磁盘核验；事实规则不代替原文语义审阅或正式质量评测。",
              "checks": {}, "passed": False}

    def check(name, passed):
        report["checks"][name] = bool(passed)

    try:
        source = ROOT / "data/raw/embedding_papers/attention.pdf"
        doc_id = hashlib.sha256(source.read_bytes()).hexdigest()
        uploaded = args.root / "raw" / doc_id / source.name
        check("上传原文与完整论文哈希一致", uploaded.exists() and uploaded.read_bytes() == source.read_bytes())
        # 禁止在路径错误时创建空库，再将它误当成成功证据。
        if not (args.root / "index/chroma.sqlite3").is_file():
            raise FileNotFoundError("测试索引不存在")
        with patch("src.retrieval.vector_store.load_config", return_value=config):
            chunks = VectorStore().list_chunks()
        expected = split_documents(load_pdf(uploaded))
        actual = {chunk.metadata["chunk_id"]: chunk for chunk in chunks}
        check("所有预期分块均持久化且正文元数据一致", len(actual) == len(expected) > 0 and all(
            chunk.metadata["chunk_id"] in actual
            and actual[chunk.metadata["chunk_id"]] == chunk for chunk in expected))
        report["ingestion"] = {"file": source.name, "sha256": doc_id, "bytes": source.stat().st_size,
                               "expected_chunks": len(expected), "stored_chunks": len(chunks),
                               "chunk_ids": sorted(actual)}
        records = [json.loads(line) for path in sorted((args.root / "logs").glob("agent_*.jsonl"))
                   for line in path.read_text().splitlines()]
        rag_records = [json.loads(line) for path in sorted((args.root / "logs").glob("rag_*.jsonl"))
                       for line in path.read_text().splitlines()]
        report.update(agent_records=records, rag_records=rag_records)
        latest = {}
        for record in records:
            latest[record["request_id"]] = record
        # 历史失败完整保存，成功条件只针对最新完整流程复测，不能抹掉旧失败。
        last = list(latest.values())[-1]
        check("最新请求正常完成", last["event"] == "done" and last["task_complete"] is True)
        report["request_outcomes"] = [{"request_id": r["request_id"], "event": r["event"],
                                       "task_complete": r["task_complete"], "stop_reason": r["stop_reason"]}
                                      for r in latest.values()]
        trace = last["metrics"]["trace"]
        phases = [step["type"] for step in trace]
        check("决策调用观察终止顺序完整", phases == ["thought", "tool_call", "tool_result", "observation", "done"])
        results = [s for s in trace if s["type"] == "tool_result"]
        check("实际调用知识库核心工具成功", len(results) == 1 and results[0]["name"] == "knowledge_base_search"
              and results[0]["status"] == "success" and results[0]["result"]["status"] == "answered")
        tool_result = (results[0].get("result") or {}) if results else {}
        citations = tool_result.get("citations", [])
        check("引用对应入库正文与页码", bool(citations) and all(
            ref["metadata"].get("chunk_id") in actual
            and all(actual[ref["metadata"]["chunk_id"]].metadata.get(k) == v for k, v in ref["metadata"].items())
            and actual[ref["metadata"]["chunk_id"]].page_content.startswith(ref["text"])
            and ref["source_file"] == source.name and 1 <= ref["metadata"].get("page_number", 0) <= 15
            for ref in citations))
        check("实际五块混合重排与本地生成日志完成", any(
            r["request_id"] == tool_result.get("retrieval", {}).get("request_id")
            and r["status"] == "completed" and r["generation_mode"] == "grounded"
            and len(r["retrieval"]["documents"]) == 5 and r["tokens"]["total"] > 0 for r in rag_records)
              if results else False)
        memory = MemoryManager(args.root / "memory.sqlite3")
        messages = memory.get_messages(last["user_id"], last["session_id"])
        answer = messages[-1].content
        check("问答及同请求展示快照保存到SQLite", len(messages) == 2
              and messages[0].content == last["question"] and messages[-1].additional_kwargs["event"]["request_id"] == last["request_id"])
        check("最终回答含6层512维及第3页来源", bool(re.search(r"6|六", answer)) and "512" in answer
              and "attention.pdf" in answer and bool(re.search(r"第\s*3\s*页", answer)))
        report["answer"] = answer
        report["passed"] = all(report["checks"].values())
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"passed": report["passed"], "checks": report["checks"], "error": report.get("error")}, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
