"""本地真实M3E／BM25／RRF／BGE确认入口复测；不模拟用户确认，不调用生成模型。"""

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
from src.agent.tools import _paper_compare
from src.chunking import split_documents
from src.data_loader import create_import_tasks, batch_import
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config

output = Path(__file__).with_name("真实方法确认入口.json")
if output.exists():
    raise FileExistsError("不覆盖历史实测，请使用新的报告目录")
config = deepcopy(load_config())
inputs = [("完整Attention.pdf", (ROOT / "data/raw/embedding_papers/attention.pdf").read_bytes()),
          ("Windows验收_ViT.pdf", (ROOT / "reports/5_4_4 端到端联调与测试/Windows浏览器验收_20261004/测试输入/Windows验收_ViT.pdf").read_bytes())]
report = {"scope": __doc__, "config": deepcopy(config),
          "inputs": [{"file": name, "sha256": sha256(data).hexdigest()} for name, data in inputs],
          "passed": False}
started = perf_counter()
try:
    # 只隔离临时路径；真实模型、解析、分块和持久化向量均沿用产品配置。
    with tempfile.TemporaryDirectory(prefix="rag-method-confirm-") as directory, ExitStack() as stack:
        for key, name in (("raw_documents", "raw"), ("vector_index", "index"), ("logs", "logs")):
            config["paths"][key] = str(Path(directory) / name)
        for module in ("src.agent.tools", "src.retrieval.vector_store", "src.retrieval.hybrid_retriever",
                       "src.retrieval.reranker", "src.generation.rag_pipeline", "src.generation.cache", "src.utils.logger"):
            stack.enter_context(patch(module + ".load_config", return_value=config))
        # 低分应暂停并展示候选；若意外开始生成，立即失败，不以mock答复冒充真实质量。
        generation = stack.enter_context(patch("src.agent.tools._tool_model_response", side_effect=AssertionError("未确认时不能生成")))
        tasks = create_import_tasks(inputs)
        list(batch_import(tasks, config["paths"]["raw_documents"]))
        store = VectorStore()
        store.add_chunks([chunk for task in tasks for chunk in split_documents(task["documents"])])
        report["chunks"] = store.count()
        print("真实入库块数", report["chunks"], flush=True)
        pending = {}
        result = _paper_compare(*(row["sha256"] for row in report["inputs"]), session_id="local-proof",
                                pending=pending, request_question="对比两篇论文的方法、数据集与实验结果")
        report["result"] = result
        assert result["status"] == "needs_confirmation"
        assert not result.get("confirmed", False)
        assert result["usage"]["eval_count"] == 0
        generation.assert_not_called()
        approval = pending[result["confirmation_id"]]
        for paper in result["papers"]:
            # 工具输出会移除内部选择标记；从已登记的确认上下文核对实际方法证据。
            method = next(ref for ref in approval["context"]["references"]
                          if ref["metadata"].get("comparison_method") and ref["metadata"]["doc_id"] == paper["doc_id"])
            assert method["text"].upper().startswith("ABSTRACT")
            assert paper["coverage"]["方法"] == {"generation_mode": "low", "top_score": method["score"]}
            displayed = next(ref for ref in paper["references"] if ref["metadata"]["chunk_id"] == method["metadata"]["chunk_id"])
            assert displayed["text"] == method["text"] and displayed["score"] == method["score"]
        report["pending"] = {key: approval[key] for key in ("tool_name", "question", "args", "session_id", "scope")}
        assert approval["base"]["papers"] == result["papers"]
        assert {"论文A：方法", "论文B：方法"} <= set(result["low_relevance_dimensions"])
        report["passed"] = True
        print("状态", result["status"], "方法分数", [paper["coverage"]["方法"]["top_score"] for paper in result["papers"]], flush=True)
except Exception as error:
    report["error"] = repr(error)
    raise
finally:
    report["elapsed_seconds"] = perf_counter() - started
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
