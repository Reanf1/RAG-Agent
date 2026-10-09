"""复测两条控制题和两条已记录失败；8条局部结果不冒充120条全量成绩。"""

import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
source_runtime, output = map(Path, sys.argv[1:3])
runtime = output.with_name(output.stem + "-runtime")
dataset = output.with_name(output.stem + "-questions.json")
if output.exists() or dataset.exists() or runtime.exists():
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
# 旧轮目录保持冻结；新题集、新源码各自冻结到独立目录，索引内容沿用副本。
runtime.mkdir()
for directory in ("index", "raw"):
    shutil.copytree(source_runtime / directory, runtime / directory)
fingerprint = {"dataset_sha256": module.digest(dataset), "manifest_sha256": module.digest(module.MANIFEST),
               "corpus_sha256": module.digest(module.CORPUS), "config_sha256": module.digest(ROOT / "config.yaml"),
               "evaluation_script_sha256": module.digest(Path(spec.origin)),
               "source_sha256": {str(p.relative_to(ROOT)): module.digest(p)
                                  for p in sorted((ROOT / "src").rglob("*.py"))}}
module.write_json(runtime / "input_fingerprint.json", fingerprint)
sys.argv = [str(spec.origin), "--stage", "agent", "--root", str(runtime),
            "--output", str(output), "--limit", "4"]
module.main()
