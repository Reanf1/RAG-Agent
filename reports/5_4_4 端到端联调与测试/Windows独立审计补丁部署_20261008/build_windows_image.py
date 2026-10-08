"""在 Windows 交互用户会话构建本轮独立镜像，保留日志；不覆盖生产镜像。"""

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    context = args.run / "build-context"
    if context.exists():
        raise FileExistsError("本轮构建目录已存在，拒绝覆盖")
    context.mkdir()
    for name in ("src", "docker"):
        shutil.copytree(args.project / name, context / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("config.yaml", "requirements.txt"):
        shutil.copy2(args.project / name, context / name)
    # 只规范隔离构建副本的 Git 换行，Windows 原项目不写入。
    for path in context.rglob("*"):
        if path.is_file() and (path.suffix in {".py", ".yaml", ".yml", ".txt"} or path.name == "Dockerfile"):
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    expected = json.loads((args.run / "源码指纹.json").read_text(encoding="utf-8"))
    checks = []
    for row in expected["files"]:
        if row["path"].startswith("tests/"):
            continue
        path = context / row["path"]
        checks.append({"path": row["path"], "passed": sha256(path.read_bytes()).hexdigest() == row["lf_sha256"]})
    assert all(row["passed"] for row in checks), "构建副本与提交不一致"
    started = datetime.now().astimezone().isoformat()
    image = "rag-agent:win-audit-20261008"
    with (args.run / "docker-build.log").open("xb") as output:
        code = subprocess.run(["docker", "build", "--platform=linux/amd64", "--progress=plain",
                               "-t", image, "-f", str(context / "docker/Dockerfile"), str(context)],
                              stdout=output, stderr=subprocess.STDOUT, check=False).returncode
    result = {"source_revision": expected["source_revision"], "started_at": started,
              "finished_at": datetime.now().astimezone().isoformat(), "image": image,
              "context": str(context), "checks": checks, "exit_code": code, "passed": code == 0}
    with (args.run / "docker-build-result.json").open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
