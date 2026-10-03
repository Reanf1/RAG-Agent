"""固定真实论文上下文，逐项比较本地 Qwen 的生成采样参数并保存全部答案。"""

import argparse
import hashlib
import json
import platform
import random
import re
import statistics
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))

GROUPS = {
    "baseline": {"temperature": 0.1, "top_p": 0.9, "top_k": 40},
    "temperature_0": {"temperature": 0.0, "top_p": 0.9, "top_k": 40},
    "temperature_04": {"temperature": 0.4, "top_p": 0.9, "top_k": 40},
    "temperature_08": {"temperature": 0.8, "top_p": 0.9, "top_k": 40},
    "top_p_06": {"temperature": 0.1, "top_p": 0.6, "top_k": 40},
    "top_p_10": {"temperature": 0.1, "top_p": 1.0, "top_k": 40},
    "top_k_10": {"temperature": 0.1, "top_p": 0.9, "top_k": 10},
    "top_k_80": {"temperature": 0.1, "top_p": 0.9, "top_k": 80},
}
SEEDS = (17, 29)
RULE_VERSION = "facts-and-body-citations-v2"


def evaluate_answer(case: dict, result: dict) -> dict:
    """仅做可重复的规则检查；事实出现不等于对应关系正确或没有幻觉。"""
    # 英文回答可能翻译来源标题；末尾编号仍不能冒充正文逐项引用。
    body = re.split(r"^##\s+(?:参考来源|Reference Sources|References|Sources)\s*$",
                    result["raw_answer"], maxsplit=1, flags=re.M | re.I)[0]
    normalized = unicodedata.normalize("NFKC", body).replace("−", "-")
    # 数字旁的汉字不应阻止 \b 匹配；ASCII 边界避免把“6层”漏算，同时不匹配 16。
    hits = [bool(re.search(pattern, normalized, re.I | re.S | re.A)) for pattern in case["fact_patterns"]]
    if case["id"] == "insufficient" and "当前知识库中未找到相关文档" in body:
        hits[0] = True  # 模板的无相关文档提示属于拒答；语义审阅另核对其是否准确。
    body_ids = {int(value) for value in re.findall(r"\[参考文档([0-9]+)\]", body)}
    actual = {item["id"] for item in result["citations"]} & body_ids
    expected = set(case["expected_citation_ids"])
    citations_ok = expected <= actual and not result["invalid_citation_ids"]
    # 无依据题不应补引用；只说明原文未包含信息时可以引已有片段。
    if case["id"] == "empty":
        citations_ok = citations_ok and not actual
    format_ok = all(heading in result["raw_answer"] for heading in ("## 回答", "## 参考来源"))
    stopped = result["done_reason"] == "stop"
    return {"fact_hits": hits, "fact_coverage": sum(hits) / len(hits), "citation_required": bool(expected),
            "expected_citations_ok": citations_ok, "format_ok": format_ok,
            "natural_stop": stopped, "rule_pass": all(hits) and citations_ok and format_ok and stopped}


def summarize(rows: list[dict]) -> dict:
    """每组两种子、同样十题等权平均；耗时不含服务启动、下载与预热。"""
    return {"answers": len(rows),
            **{key: statistics.mean(row["checks"][key] for row in rows) for key in
               ("fact_coverage", "expected_citations_ok", "format_ok", "natural_stop", "rule_pass")},
            "factual_key_coverage": statistics.mean(row["checks"]["fact_coverage"] for row in rows
                                                   if row["checks"]["citation_required"]),
            "body_citation_rate": statistics.mean(row["checks"]["expected_citations_ok"] for row in rows
                                                  if row["checks"]["citation_required"]),
            "refusal_notice_rate": statistics.mean(row["checks"]["fact_coverage"] for row in rows
                                                   if not row["checks"]["citation_required"]),
            "wall_seconds_mean": statistics.mean(row["wall_seconds"] for row in rows),
            "output_tokens_mean": statistics.mean(row["result"]["usage"]["eval_count"] for row in rows),
            "by_seed": {str(seed): {"rule_pass": statistics.mean(row["checks"]["rule_pass"]
                                   for row in rows if row["seed"] == seed)} for seed in SEEDS}}


def main():
    """先锁定输入和配置，再预热、交错执行，逐条写盘；失败立即停止而非补假答案。"""
    from src.generation.prompt_template import PROMPT_VERSION, build_rag_messages
    from src.generation.rag_pipeline import generate_answer
    from src.utils.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/5_2_1 Prompt工程与生成策略/生成参数对比结果.json")
    parser.add_argument("--rescore", action="store_true", help="只重算已有原始答案的规则分数，不调用模型")
    args = parser.parse_args()
    sample_path = PROJECT_ROOT / "reports/5_2_1 Prompt工程与生成策略/生成参数评测集.json"
    output = args.output
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    if args.rescore:
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "completed", "不能重算仍在写入的实验结果"
        assert report["dataset_sha256"] == hashlib.sha256(sample_path.read_bytes()).hexdigest()
        cases = {case["id"]: case for case in sample["cases"]}
        for row in report["rows"]:
            row["checks"] = evaluate_answer(cases[row["case_id"]], row["result"])
        report["summary"] = {group: summarize([row for row in report["rows"] if row["group"] == group])
                             for group in GROUPS}
        report["rule_version"] = RULE_VERSION
        report["rescored_at"] = datetime.now().astimezone().isoformat()
        report["scoring_note"] = "统一修正中文数字边界、无相关文档拒答及英文来源区识别；问题、Prompt、上下文和原始生成答案均未修改。"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return
    assert sample["prompt_version"] == PROMPT_VERSION, "Prompt 版本变化，请重新核对输入"
    assert len({case["id"] for case in sample["cases"]}) == len(sample["cases"])
    for case in sample["cases"]:
        messages = [{"role": "user" if message.type == "human" else message.type,
                     "content": message.content}
                    for message in build_rag_messages(case["question"], case["context"]["context"])]
        digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        assert digest == case["messages_sha256"], "问题、上下文或实际模板变化"
    config = load_config()["llm"]
    base_url = config["base_url"].rstrip("/")
    with urlopen(base_url + "/api/version", timeout=10) as response:
        server_version = json.load(response)
    with urlopen(base_url + "/api/tags", timeout=10) as response:
        tags = json.load(response)
    tag = next(model for model in tags["models"] if model["name"] == config["model"])
    request = Request(base_url + "/api/show", data=json.dumps({"model": config["model"]}).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        details = json.load(response)
    if output.exists():
        raise FileExistsError("结果文件已存在；复测使用 --output 指定新路径，避免覆盖真实记录")
    report = {"started_at": datetime.now().astimezone().isoformat(), "status": "running",
              "scope": "固定标注相关片段的生成控制实验，不运行检索、不计算检索指标",
              "environment": {"platform": platform.platform(), "python": platform.python_version()},
              "server": server_version, "model": tag, "model_parameters": details.get("parameters"),
              "model_info": details.get("model_info"), "llm_config": config, "prompt_version": PROMPT_VERSION,
              "dataset_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
              "groups": GROUPS, "seeds": SEEDS, "order_seed": 20260929,
              "rule_version": RULE_VERSION, "rows": []}

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    warmup = generate_answer(sample["cases"][0]["question"], sample["cases"][0]["context"],
                             options={**GROUPS["baseline"], "seed": SEEDS[0]})
    report["warmup"] = warmup
    with urlopen(base_url + "/api/ps", timeout=10) as response:
        report["loaded_model"] = json.load(response)
    save()
    jobs = [(group, seed, case) for seed in SEEDS for case in sample["cases"] for group in GROUPS]
    random.Random(report["order_seed"]).shuffle(jobs)
    for index, (group, seed, case) in enumerate(jobs, 1):
        started = perf_counter()
        result = generate_answer(case["question"], case["context"], options={**GROUPS[group], "seed": seed})
        row = {"order": index, "group": group, "seed": seed, "case_id": case["id"],
               "wall_seconds": perf_counter() - started, "result": result,
               "checks": evaluate_answer(case, result)}
        report["rows"].append(row)
        save()
        print(f"{index}/{len(jobs)} {group} seed={seed} {case['id']} "
              f"规则={'通过' if row['checks']['rule_pass'] else '需核对'} "
              f"{row['wall_seconds']:.2f}s", flush=True)
    report["summary"] = {group: summarize([row for row in report["rows"] if row["group"] == group])
                         for group in GROUPS}
    report["status"] = "completed"
    report["finished_at"] = datetime.now().astimezone().isoformat()
    save()
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
