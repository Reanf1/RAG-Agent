"""使用真实本地 Qwen 记录片段、正文/引用出现时序，保留可复核的原始回答。"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.prompt_template import PROMPT_VERSION
from src.generation.rag_pipeline import urlopen
from src.generation.streaming import stream_answer
from src.utils.config import load_config


def measure(question: str, context: dict) -> dict:
    """首片段与首可见正文分别计时；只把 done 包的计数当作实际 Token。"""
    started = perf_counter()
    packets, final = [], None
    first_text, first_citation = None, None
    for event in stream_answer(question, context, options={"seed": 17}):
        elapsed = perf_counter() - started
        if event["type"] == "token":
            packets.append({"seconds": elapsed, "content": event["content"], "answer": event["answer"],
                            "citation_ids": [ref["id"] for ref in event["citations"]]})
            if first_text is None and event["answer"].strip():
                first_text = elapsed
            if first_citation is None and event["citations"]:
                first_citation = elapsed
        elif event["type"] == "error":
            raise RuntimeError(event["message"])
        else:
            final = event
    if final is None:
        raise RuntimeError("流未返回 done")
    total = perf_counter() - started
    assert "".join(packet["content"] for packet in packets) == final["raw_answer"]
    assert packets[0]["seconds"] < total
    if first_citation is not None:
        assert first_citation < total
    return {"question": question, "context": context, "packets": packets, "result": final,
            "timing": {"first_packet_seconds": packets[0]["seconds"], "first_text_seconds": first_text,
                       "first_citation_seconds": first_citation, "done_seconds": total,
                       "packet_count": len(packets)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/5_2_2 流式输出与引用/流式与引用验证结果.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("请指定新结果路径，保留已有实测记录")
    dataset_path = PROJECT_ROOT / "reports/5_2_1 Prompt工程与生成策略/生成参数评测集.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset["cases"][:2]
    # 两题复用既有真实论文块，新增英文引用请求单独标记，不冒充参数实验原题。
    cases.append({**cases[1], "id": "bert-sizes-inline",
                  "question": cases[1]["question"] + " Cite [参考文档1] immediately after each factual sentence."})
    pdfs = []
    for paper in dataset["papers"]:
        path = PROJECT_ROOT / "data/raw/embedding_papers" / (paper["id"] + ".pdf")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == paper["pdf_sha256"]
        pdfs.append({"path": str(path), "sha256": digest})
    config = load_config()
    base_url = config["llm"]["base_url"]
    report = {"started_at": datetime.now().astimezone().isoformat(),
              "scope": "真实本地生成及增量引用验证；固定上下文，不是端到端检索质量/独立人工评分。",
              "environment": {"python": platform.python_version(), "platform": platform.platform()},
              "server": json.load(urlopen(base_url + "/api/version", timeout=10)),
              "models": json.load(urlopen(base_url + "/api/tags", timeout=10)),
              "llm_config": config["llm"], "prompt_version": PROMPT_VERSION,
              "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
              "verified_pdfs": pdfs, "rows": []}
    for case in cases:
        row = {"id": case["id"], **measure(case["question"], case["context"])}
        report["rows"].append(row)
        print(case["id"], json.dumps(row["timing"], ensure_ascii=False), flush=True)
    assert any(row["timing"]["first_citation_seconds"] is not None for row in report["rows"])
    report["loaded_models"] = json.load(urlopen(base_url + "/api/ps", timeout=10))
    report["finished_at"] = datetime.now().astimezone().isoformat()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
