"""核对部署文件与本轮 Git 源码指纹；允许 Git 的 LF/CRLF 转换，保存实际字节指纹。"""

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("不覆盖已有部署核验记录")
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.project, text=True
    ).strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected["source_revision"], revision],
        cwd=args.project, check=False,
    ).returncode == 0
    checks = []
    for row in expected["files"]:
        path = args.project / row["path"]
        actual = path.read_bytes() if path.is_file() else None
        # 源码都是文本。只统一换行，不忽略内容、空白或编码差异。
        normalized = sha256(actual.replace(b"\r\n", b"\n")).hexdigest() if actual is not None else None
        checks.append({
            "path": row["path"], "expected_lf_sha256": row["lf_sha256"],
            "actual_sha256": sha256(actual).hexdigest() if actual is not None else None,
            "actual_lf_sha256": normalized, "passed": normalized == row["lf_sha256"],
        })
    result = {
        "checked_at": datetime.now().astimezone().isoformat(), "revision": revision,
        "source_revision": expected["source_revision"], "source_is_ancestor": ancestor,
        "checks": checks, "passed": ancestor and all(row["passed"] for row in checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({"revision": revision, "files": len(checks), "passed": result["passed"]}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
