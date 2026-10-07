"""独立进程复核原库正文、来源及原文文件指纹，测试不改写原文或块内容。"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("不覆盖已有原库证据")
    sys.path.insert(0, str(args.project))
    from src.retrieval.repair_index import content_digest
    from src.retrieval.vector_store import VectorStore
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    store = VectorStore(expected["source_index"])
    chunks = store.list_chunks()
    digest = content_digest(chunks)
    files = []
    for row in expected["files"]:
        path = Path(expected["source_raw"]) / row["doc_id"] / row["file"]
        files.append({**row, "unchanged": sha256(path.read_bytes()).hexdigest() == row["sha256"]})
    report = {"chunks": len(chunks), "documents": len({doc.metadata["doc_id"] for doc in chunks}),
              "content_sha256": digest, "files": files,
              "passed": digest == expected["source_content_sha256"] and all(row["unchanged"] for row in files)}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    assert report["passed"]


if __name__ == "__main__":
    main()
