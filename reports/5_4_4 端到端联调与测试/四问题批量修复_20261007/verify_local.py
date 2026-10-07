"""四问题的本地真实模型复测；不连接Windows，不作部署／离线／性能验收。"""

from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.agent.tools import _knowledge_base_search, paper_compare, _expand_paper_evidence
from src.chunking import split_documents
from src.data_loader import create_import_tasks, batch_import, load_document
from src.generation.rag_pipeline import build_context, generate_answer
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config

OUTPUT = Path(__file__).with_name("真实模型结果.json")
if OUTPUT.exists():
    raise FileExistsError("不覆盖历史实测，请将既有输出归档后重跑")
report = {"scope": __doc__, "cases": [], "complete": False}


def save():
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(name, callback):
    start = perf_counter()
    try:
        result = callback()
        report["cases"].append({"name": name, "seconds": perf_counter() - start, "result": result})
        print(name, result.get("status"), result.get("answer"), flush=True)
    except Exception as error:
        report["cases"].append({"name": name, "seconds": perf_counter() - start, "error": repr(error)})
        print(name, repr(error), flush=True)
    save()


config = deepcopy(load_config())
report["config"] = config
pdf = ROOT / "reports/5_4_4 端到端联调与测试/Windows浏览器验收_20261004/测试输入/Windows验收_ViT.pdf"
inputs = [(pdf.name, pdf.read_bytes()), ("完整Attention.pdf", (ROOT / "data/raw/embedding_papers/attention.pdf").read_bytes()),
          ("DETR节选.pdf", pdf.with_name("Windows验收_DETR节选.pdf").read_bytes())]
report["inputs"] = [{"file": name, "sha256": sha256(data).hexdigest()} for name, data in inputs]
# 临时路径隔离用户知识库与会话；检索、权重和HTTP生成均真实运行。
with tempfile.TemporaryDirectory(prefix="rag-batch-local-") as directory, ExitStack() as stack:
    for key, name in (("raw_documents", "raw"), ("vector_index", "index"), ("logs", "logs")):
        config["paths"][key] = str(Path(directory) / name)
    for module in ("src.agent.tools", "src.retrieval.vector_store", "src.retrieval.hybrid_retriever",
                   "src.retrieval.reranker", "src.generation.rag_pipeline", "src.utils.logger"):
        stack.enter_context(patch(module + ".load_config", return_value=config))
    tasks = create_import_tasks(inputs)
    list(batch_import(tasks, config["paths"]["raw_documents"]))
    store = VectorStore()
    chunks = []
    for task in tasks:
        chunks.extend(split_documents(task["documents"]))
    store.add_chunks(chunks)
    report["chunks"] = store.count()
    save()
    vit_id, attention_id, detr_id = [row["sha256"] for row in report["inputs"]]
    # 修复前页面选中的真实两个块原样生成，记录相邻分类头带来的风险；不把它当必然复现。
    previous = [(doc, score) for prefix, score in (("de306f23", .9242377281), ("8c8caa8c", .8842619061))
                for doc in chunks if doc.metadata["doc_id"] == vit_id and doc.metadata["chunk_id"].startswith(prefix)]
    q = "ViT如何使用位置编码？请用中文简述并注明原文页码。"
    run("宽上下文对照", lambda: generate_answer(q, build_context(q, previous), options={"seed": 17}))
    for name, question in (("聚焦短问题", q), ("聚焦短问题重复", q),
                           ("聚焦带文档ID问题", f"请根据已入库论文{vit_id}回答：{q}")):
        run(name, lambda question=question: _knowledge_base_search(question, doc_id=vit_id))
    run("完整Attention与ViT对比", lambda: paper_compare.invoke({"paper_a_id": attention_id, "paper_b_id": vit_id}))
    run("ViT与DETR两页节选对比", lambda: paper_compare.invoke({"paper_a_id": vit_id, "paper_b_id": detr_id}))
report["complete"] = True
save()
