"""只核验8条定向复测的输入、实际调用和计量，不自动判断答案质量。"""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

root, output_dir = map(Path, sys.argv[1:3])
digest = lambda path: sha256(path.read_bytes()).hexdigest()
result = json.loads((output_dir / "targeted-agent.json").read_text(encoding="utf-8"))
fingerprint = json.loads((output_dir / "targeted-agent-runtime/input_fingerprint.json").read_text(encoding="utf-8"))
requests = [json.loads(line) for line in (output_dir / "targeted-agent.calls.jsonl").read_text(encoding="utf-8").splitlines()]
assert result["status"] == "completed" and result["question_count"] == 4
assert result["inputs"] == fingerprint
expected = {(qid, profile) for qid in ("F010", "F012", "C013", "S005") for profile in ("default", "no_rules")}
assert Counter((r["id"], r["profile"]) for r in result["rows"]) == Counter(expected)
for key, path in {
    "dataset_sha256": output_dir / "targeted-agent-questions.json",
    "manifest_sha256": root / "reports/5_5_1 评测集构建/论文清单.json",
    "corpus_sha256": root / "data/raw/evaluation_vision_transformers/corpus.json",
    "config_sha256": root / "config.yaml",
    "evaluation_script_sha256": root / "reports/5_5_2 系统性能评估/evaluate_system.py",
}.items():
    assert digest(path) == fingerprint[key], (key, path)
for name, expected_sha in fingerprint["source_sha256"].items():
    assert digest(root / name) == expected_sha, name
for row in result["rows"]:
    calls = [call for call in requests if (call["id"], call["profile"]) == (row["id"], row["profile"])]
    assert len(calls) == row["model_calls"]
    assert all(call["request"]["options"]["seed"] == result["model_seed"] == 20261003 for call in calls)
    input_tokens = sum(call["response"]["prompt_eval_count"] for call in calls)
    output_tokens = sum(call["response"]["eval_count"] for call in calls)
    assert row["tokens"]["input_known"] == input_tokens
    assert row["tokens"]["output_known"] == output_tokens
    assert row["tokens"]["total"] == input_tokens + output_tokens
    assert row["tokens"]["unknown_calls"] == 0
summary = {
    "scope": "8条定向复测的证据核验；不是120条完整质量重评或人工评分",
    "status": "passed", "records": len(result["rows"]), "actual_http_calls": len(requests),
    "source_files_verified": len(fingerprint["source_sha256"]),
    "input_fingerprint_verified": True, "actual_tokens_verified": True,
    "stop_reasons": dict(Counter(call["response"]["done_reason"] for call in requests)),
    "tool_execution_errors": sum(event["status"] != "success" for row in result["rows"]
                                  for event in row["metrics"]["trace"] if event["type"] == "tool_result"),
    "evidence_sha256": {name: digest(output_dir / name) for name in (
        "run.json", "targeted-agent.json", "targeted-agent.calls.jsonl", "targeted-agent-questions.json",
        "targeted-agent-runtime/input_fingerprint.json", "regression.json", "parallel-after.json")},
}
(output_dir / "targeted-verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
