"""真实本地 Qwen 的 Thought 单轮规划开发验证，不执行工具或完整ReAct循环。"""

import argparse
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from langchain_core.tools import tool

from src.agent.react_loop import think
from src.generation.rag_pipeline import urlopen
from src.utils.config import load_config
from src.utils.logger import request_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/5_3_1 Agent核心循环/Thought单轮规划验证结果_20261002.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("请指定不存在的新结果路径，保留历史实测记录")
    invocations = []

    @tool
    def multiply(a: float, b: float) -> float:
        """精确计算两个数字的乘积。"""
        invocations.append((a, b))
        return a * b

    config = load_config()
    cases = [
        {"id": "concept", "question": "什么是过拟合？", "tools": [], "context": {}, "expected": "answer"},
        {"id": "calculate", "question": "3.14乘以2.56是多少？", "tools": [multiply], "context": {}, "expected": "tool"},
        {"id": "after_success", "question": "3.14乘以2.56是多少？", "tools": [multiply],
         "context": {"observations": [{"tool_name": "multiply", "result": "3.14 × 2.56 = 8.0384", "status": "success"}]},
         "expected": "answer"},
        {"id": "paper_without_search", "question": "已上传论文A使用了什么数据集？", "tools": [multiply], "context": {}, "expected": "answer"},
    ]
    report = {"started_at": request_time(), "config": config, "python": platform.python_version(),
              "server": json.load(urlopen(config["llm"]["base_url"] + "/api/version", timeout=10)),
              "models": json.load(urlopen(config["llm"]["base_url"] + "/api/tags", timeout=10)),
              "scope": "四个开发样例的真实LLM单轮计划；multiply仅为测试描述，成功观察人工构造，不计为生产工具、实际Action或路由评测。",
              "validation_complete": False, "rows": []}
    for case in cases:
        result = think(case["question"], case["tools"], case["context"])
        row = {"id": case["id"], "question": case["question"], "context": case["context"],
               "available_tools": [item.name for item in case["tools"]], "expected_next_step": case["expected"],
               "result": result, "expected_step_match": result["next_step"] == case["expected"]}
        report["rows"].append(row)
        report["tool_execution_count"] = len(invocations)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        assert not invocations and row["expected_step_match"]
        assert result["usage"]["prompt_eval_count"] > 0 and result["usage"]["eval_count"] > 0
        print(case["id"], result["thought"], result["next_step"], result["tool_name"], flush=True)
    report["validation_complete"], report["finished_at"] = True, request_time()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
