"""复测两条控制题和两条已记录失败；8条局部结果不冒充120条全量成绩。"""

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
runtime, output = map(Path, sys.argv[1:3])
dataset = output.with_name(output.stem + "-questions.json")
if output.exists() or dataset.exists():
    raise FileExistsError("局部复测使用新输出，不覆盖失败证据")
questions = json.loads((ROOT / "reports/评测集.json").read_text(encoding="utf-8"))
selected = [q for q in questions if q["id"] in {"F010", "F012", "C013", "S005"}]
assert len(selected) == 4
dataset.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
spec = importlib.util.spec_from_file_location("evaluate_system", ROOT / "reports/5_5_2 系统性能评估/evaluate_system.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# 只替换本次评测输入；原60题、业务配置与提示词不改写。
module.DATASET = dataset
sys.argv = [str(spec.origin), "--stage", "agent", "--root", str(runtime),
            "--output", str(output), "--limit", "4"]
module.main()
