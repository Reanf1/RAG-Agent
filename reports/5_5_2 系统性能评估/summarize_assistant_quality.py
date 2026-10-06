"""展开人工阅读后的助手判断并核验180条记录；不使用算法打质量分。"""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROFILES = ("default", "no_rules", "optimized")
source = json.loads((HERE / "人工评分180条来源.json").read_text(encoding="utf-8"))
questions = json.loads((ROOT / "reports/评测集.json").read_text(encoding="utf-8"))
judgment = json.loads((HERE / "助手逐题判断_20261006.json").read_text(encoding="utf-8"))
evidence = json.loads((HERE / "助手初评原文证据_20261006.json").read_text(encoding="utf-8"))
qa = {row["id"]: row for row in questions}
notes = {row["id"]: row for row in judgment["逐题判断"]}
assert len(notes) == len(qa) == 60 and set(notes) == set(qa)
assert len(source["rows"]) == 180
assert {(r["id"], r["profile"]) for r in source["rows"]} == {
    (identifier, profile) for identifier in qa for profile in PROFILES}
rows = []
for answer in source["rows"]:
    identifier, profile = answer["id"], answer["profile"]
    note, question = notes[identifier], qa[identifier]
    scores = note["scores"][PROFILES.index(profile)]
    assert len(scores) == 3 and all(type(s) is int and 0 <= s <= 4 for s in scores)
    assert note["reason"].strip()
    rows.append({"id": identifier, "profile": profile, "category": question["category"],
                 "question": question["question"], "answer": answer["answer"],
                 "answer_sha256": sha256(answer["answer"].encode("utf-8")).hexdigest(),
                 "reference_answer": question["reference_answer"],
                 "answer_points": question["answer_points"], "evidence": question["evidence"],
                 "assistant_scores": dict(zip(("correctness", "completeness", "citation_accuracy"), scores)),
                 "assistant_reason": note["reason"],
                 "user_review": {"correctness": None, "completeness": None,
                                 "citation_accuracy": None, "reason": None,
                                 "reviewer": None, "date": None}})
summary = {}
for profile in PROFILES:
    selected = [row for row in rows if row["profile"] == profile]
    summary[profile] = {"count": len(selected), "user_reviewed": 0,
                        "means": {key: sum(r["assistant_scores"][key] for r in selected) / len(selected)
                                  for key in selected[0]["assistant_scores"]},
                        "core_unanswered_or_wrong": sum(r["assistant_scores"]["correctness"] == 0 for r in selected),
                        "without_valid_citation": sum(r["assistant_scores"]["citation_accuracy"] == 0 for r in selected)}
files = [HERE / "人工评分180条来源.json", ROOT / "reports/评测集.json",
         ROOT / "reports/5_5_1 评测集构建/论文清单.json", HERE / "评测方案.json",
         HERE / "助手逐题判断_20261006.json", HERE / "助手初评原文证据_20261006.json"]
result = {"reviewer": judgment["评阅者"], "date": judgment["评阅日期"],
          "scope": "2026-10-03历史运行180条最终答案；不是当前代码重新运行，也不是独立人工终审",
          "method": judgment["口径"], "source_fingerprints": {
              str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in files},
          "pdf_evidence_pages": len(evidence["pages"]), "summary": summary, "rows": rows}
out = HERE / "助手初评180条_20261006.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
counts = Counter(r["category"] for r in rows)
assert all(counts[c] == 45 for c in ("fact", "comparison", "synthesis", "reasoning"))
print(json.dumps({"records": len(rows), "categories": dict(counts), "summary": summary}, ensure_ascii=False, indent=2))
