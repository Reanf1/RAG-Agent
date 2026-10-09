"""保留原生会话基线，核对原文和索引；报告不导出用户历史正文。"""
from pathlib import Path
import argparse
import json
import sqlite3
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--phase", choices=["before", "after"], required=True)
parser.add_argument("--output", type=Path, help="复核可另存新文件，不覆盖旧证据")
args = parser.parse_args()
sys.path.insert(0, str(args.project))
from src.retrieval.repair_index import content_digest
from src.retrieval.vector_store import VectorStore
from hashlib import sha256

backup = args.run / "original-memory.sqlite3"
database = args.project / "data/sessions/memory.sqlite3"
current = sqlite3.connect(str(database))
if args.phase == "before":
    if backup.exists():
        raise FileExistsError("不覆盖会话基线")
    target = sqlite3.connect(str(backup))
    current.backup(target)
    target.close()
original = sqlite3.connect(str(backup))
rows = []
for table in ("sessions", "messages", "summaries", "rag_history"):
    left = set(original.execute("SELECT * FROM " + table).fetchall())
    right = set(current.execute("SELECT * FROM " + table).fetchall())
    rows.append(dict(table=table, before=len(left), after=len(right), preserved=left.issubset(right)))
current.close()
original.close()
chunks = VectorStore(args.project / "data/index").list_chunks()
files = [{"path": str(path.relative_to(args.project / "data/raw")),
          "sha256": sha256(path.read_bytes()).hexdigest()}
         for path in sorted((args.project / "data/raw").glob("*/*"))
         if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}]
result = dict(phase=args.phase, sessions=rows, files=files, chunks=len(chunks),
              documents=len({d.metadata["doc_id"] for d in chunks}), content_sha256=content_digest(chunks))
if args.phase == "after":
    before = json.loads((args.run / "preserve-before.json").read_text(encoding="utf-8"))
    result["originals_unchanged"] = all(result[k] == before[k] for k in ("files", "chunks", "documents", "content_sha256"))
result["passed"] = all(row["preserved"] for row in rows) and result.get("originals_unchanged", True)
with (args.output or args.run / ("preserve-" + args.phase + ".json")).open("x", encoding="utf-8") as output:
    json.dump(result, output, ensure_ascii=False, indent=2)
print(json.dumps({key: value for key, value in result.items() if key != "files"}, ensure_ascii=False))
raise SystemExit(0 if result["passed"] else 1)
