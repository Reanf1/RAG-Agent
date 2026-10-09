"""只在独立评测索引上复现两线程冲突，保存真实异常堆栈，不调用生成模型。"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sys
from threading import Barrier
import traceback

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(4)
import src.utils.config as config_module

config = config_module.load_config()
config["paths"]["vector_index"] = str(Path(sys.argv[1]).resolve())
config_module.load_config = lambda: deepcopy(config)
from src.retrieval.vector_store import VectorStore, get_embeddings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker, get_reranker

output = Path(sys.argv[2])
if output.exists():
    raise FileExistsError("复现结果不能覆盖")
papers = json.loads((ROOT / "reports/5_5_1 评测集构建/论文清单.json").read_text(encoding="utf-8"))["papers"]
targets = [next(p["doc_id"] for p in papers if p["id"] == name) for name in ("deit", "cait")]
store = VectorStore()
retriever = HybridRetriever(store)
question = "How do DeiT and CaiT add mechanisms for producing the image-level prediction?"
get_embeddings()
get_reranker()
candidates = [retriever.search(question, k=20, doc_id=target) for target in targets]
rows = []
for stage, repeats in (("collection_read", 16), ("embedding", 4), ("rerank", 4), ("hybrid", 4)):
    for trial in range(repeats):
        barrier = Barrier(2)

        def run(index):
            barrier.wait()
            try:
                if stage == "collection_read":
                    store._store.get(where={"doc_id": targets[index]}, include=["documents", "metadatas", "embeddings"])
                elif stage == "embedding":
                    get_embeddings().embed_query(question + str(index))
                elif stage == "rerank":
                    Reranker().rerank(question, candidates[index])
                else:
                    retriever.search(question, doc_id=targets[index], rerank=True)
                return {"stage": stage, "trial": trial, "thread": index, "status": "success"}
            except Exception:
                return {"stage": stage, "trial": trial, "thread": index, "status": "error", "traceback": traceback.format_exc()}

        with ThreadPoolExecutor(2) as pool:
            rows.extend(pool.map(run, range(2)))
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(stage, trial, [row["status"] for row in rows[-2:]], flush=True)
