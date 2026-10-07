"""真实M3E与产品缓存验证条件变更；固定答案为协议样例，不作论文或生成质量成绩。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.generation.cache import SemanticCache, _constraints
from src.generation.rag_pipeline import resolve_citations
from src.retrieval.vector_store import get_embeddings
from src.utils.config import load_config


PAIRS = [
    (False, "模型准确率是否超过80%？", "模型准确率是否低于80%？"),
    (False, "模型准确率大于80%吗？", "模型准确率小于80%吗？"),
    (False, "BERT使用15%的掩码比例吗？", "BERT使用15‰的掩码比例吗？"),
    (False, "指标是否>=80？", "指标是否<=80？"),
    (False, "指标至少80吗？", "指标至多80吗？"),
    (False, "Is accuracy greater than 80%?", "Is accuracy less than 80%?"),
    (False, "指标为-5吗？", "指标为5吗？"),
    (True, "ViT采用什么输入？", "ViT使用什么输入？"),
    (True, "模型准确率是否超过80%？", "模型的准确率是否超过80%？"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("不覆盖历史结果，请使用新路径")
    report = {"scope": __doc__, "started_at": datetime.now().astimezone().isoformat(),
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "embedding": load_config()["embedding"], "cache": load_config()["generation"]["cache"],
              "pairs": [], "passed": False}
    context = {"references": [{"id": 1, "source_file": "缓存协议样例.md", "location": "行1",
               "text": "这是固定缓存协议样例，不是论文实验结果。", "metadata": {}, "truncated": False}]}
    result = {"type": "done", **resolve_citations("固定缓存答案。[参考文档1]", context),
              "done_reason": "stop", "generation_mode": "grounded", "usage": {"eval_count": 0}}
    model = get_embeddings()
    try:
        for expected, left, right in PAIRS:
            cache = SemanticCache()
            cache.put(left, result, "isolated-cache-proof")
            hit = cache.lookup(right, "isolated-cache-proof")
            a, b = [model.embed_query(question) for question in (left, right)]
            similarity = sum(x * y for x, y in zip(a, b)) / (sum(x * x for x in a) * sum(y * y for y in b)) ** .5
            row = {"cached_question": left, "query": right, "cosine": similarity,
                   "constraints_match": _constraints(left) == _constraints(right),
                   "expected_hit": expected, "actual_hit": hit is not None,
                   "cache_result": hit.get("cache") if hit else None, "passed": (hit is not None) == expected}
            report["pairs"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        report["passed"] = all(row["passed"] for row in report["pairs"])
    finally:
        report["finished_at"] = datetime.now().astimezone().isoformat()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise AssertionError("缓存条件变化或同义问句验证失败，真实结果已保留")


if __name__ == "__main__":
    main()
