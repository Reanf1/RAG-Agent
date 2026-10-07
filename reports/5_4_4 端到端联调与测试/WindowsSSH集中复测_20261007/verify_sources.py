"""从Windows原文直接核对本轮答案事实，并渲染实际物理页供人工初评。"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import fitz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("不覆盖原文核验记录")
    original = json.loads(args.expected.read_text(encoding="utf-8"))
    facts = {
        "attention.pdf": {1: ["based solely on attention", "28.4", "41.8", "eight GPUs"],
                          7: ["4.5 million", "36M"]},
        "Windows验收_ViT.pdf": {1: ["sequences of image patches"],
                            2: ["88.55", "90.72", "94.55", "77.63"],
                            4: ["1.3M", "14M", "303M"]},
    }
    checks, pages = [], []
    for row in original["files"]:
        path = Path(original["source_raw"]) / row["doc_id"] / row["file"]
        checks.append({"file": row["file"], "check": "原文指纹", "passed": sha256(path.read_bytes()).hexdigest() == row["sha256"]})
        if path.name == "Windows复测_说明.txt":
            checks.append({"file": path.name, "location": "行2", "fact": "WINCHECK20261005",
                           "passed": "WINCHECK20261005" in path.read_text(encoding="utf-8").splitlines()[1]})
        if path.name not in facts:
            continue
        with fitz.open(path) as document:
            for number, expected in facts[path.name].items():
                page = document[number - 1]
                text = " ".join(page.get_text().split())
                for fact in expected:
                    checks.append({"file": path.name, "physical_page": number, "fact": fact, "passed": fact in text})
                name = ("Attention" if path.name == "attention.pdf" else "ViT") + f"_物理页{number}.png"
                destination = args.output.parent / name
                if destination.exists():
                    raise FileExistsError("不覆盖原文渲染")
                page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4)).save(destination)
                pages.append({"file": path.name, "physical_page": number, "text": text, "render": name})
    report = {"scope": __doc__, "checks": checks, "pages": pages, "passed": all(row["passed"] for row in checks)}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": len(checks), "passed": report["passed"]}, ensure_ascii=False))
    assert report["passed"]


if __name__ == "__main__":
    main()
