"""回放真实误完成记录，只隔离Observation HTTP，不自动评价科研答案质量。"""

from io import BytesIO
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.agent.react_loop import observe
from src.agent.tools import get_available_tools

questions = {q["id"]: q for q in json.loads((ROOT / "reports/评测集.json").read_text(encoding="utf-8"))}
agent = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
checks = []
for case, profile in (("F006", "default"), ("C004", "no_rules"), ("C013", "default"),
                      ("S005", "default"), ("S005", "no_rules"), ("S009", "no_rules"), ("R012", "no_rules")):
    row = next(r for r in agent["rows"] if r["id"] == case and r["profile"] == profile)
    # 保留实际工具结果，复现模型仍然误报完成的响应，核验业务校验能否阻止它。
    observations = [{k: v for k, v in event.items() if k != "type"}
                    for event in row["metrics"]["trace"] if event["type"] == "tool_result"]
    packet = {"model": "qwen2.5:7b", "done": True, "done_reason": "stop", "message": {
        "content": json.dumps({"observation": "工具已返回，任务完成。", "decision": "finish",
                               "task_complete": True, "answer": row["answer"]}, ensure_ascii=False)},
              "prompt_eval_count": 1, "eval_count": 1}
    with patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(packet).encode("utf-8"))):
        result = observe(questions[case]["question"], get_available_tools(), {"observations": observations})
    checks.append({"id": case, "profile": profile, "task_complete": result["task_complete"], "answer": result["answer"]})
print(json.dumps(checks, ensure_ascii=False, indent=2))
assert all(not row["task_complete"] for row in checks), "无正文证据或只提取问题关键词仍被标记完成"
