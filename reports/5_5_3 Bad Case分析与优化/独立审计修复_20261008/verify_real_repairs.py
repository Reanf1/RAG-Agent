"""只读独立审计资料，以真实本地模型复测；新结果保存到独立目录。"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".test-tmp/audit-repairs/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def save(path, value):
    """不覆盖已有迭代证据；续跑请给新的输出目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, default=str)


def numerical(output):
    """实际M3E相似度与缓存命中、实际BGE末行评分，分开记录结构与质量证据。"""
    from src.retrieval.vector_store import get_embeddings
    from src.retrieval.reranker import Reranker, get_reranker
    from src.generation.cache import SemanticCache
    from src.generation.rag_pipeline import build_context
    from langchain_core.documents import Document
    embedding = get_embeddings()
    pairs = [("在一百个训练样本的设置下，实验的准确率是多少？", "在二百个训练样本的设置下，实验的准确率是多少？"),
             ("哪些模型没有使用有标签数据？", "哪些模型没有使用无标签数据？")]
    rows = []
    for first, second in pairs:
        vectors = [embedding.embed_query(text) for text in (first, second)]
        similarity = sum(a*b for a, b in zip(*vectors)) / ((sum(x*x for x in vectors[0])*sum(x*x for x in vectors[1]))**.5)
        cache = SemanticCache()
        result = {"type": "done", "done_reason": "stop", "answer": "缓存结构探针，非科研答案", "generation_mode": "grounded", "citations": [{"id": 1}], "usage": {"eval_count": 12}}
        cache.put(first, result, "probe")
        rows.append({"first": first, "second": second, "actual_m3e_cosine": similarity,
                     "changed_hit": cache.lookup(second, "probe"), "exact_hit": cache.lookup(first, "probe") is not None})
    save(output / "真实M3E缓存.json", rows)
    query = "Which model has accuracy 99.7?"
    text = "Table 1. Synthetic tail visibility probe\n| Model | Accuracy |\n| --- | --- |\n" + "".join(f"| Model-{i} | {i}.7 |\n" for i in range(100))
    document = Document(page_content=text, metadata={"content_type": "table", "source_file": "合成长表探针", "chunk_id": "tail-probe"})
    ranker, model = Reranker(), get_reranker()
    windows = ranker._passages(query, document, model.tokenizer)
    result = ranker.rerank(query, [(document, 0)], k=1)
    context = build_context(query, result)
    save(output / "真实BGE长表.json", {"boundary": "合成长表与真实BGE；只证明末尾可见及窗口长度，非真实语料总体召回率",
         "windows": windows, "pair_tokens": [len(model.tokenizer(query, window)["input_ids"]) for window in windows],
         "selected_score": result[0][1], "selected_excerpt": result[0][0].metadata.get("rerank_excerpt"),
         "tail_in_prompt": "Model-99" in context["context"], "original_unchanged": "rerank_excerpt" not in document.metadata})


def observe_http(output):
    """包装真实响应的read/迭代，只记录，不替换模型内容与业务逻辑。"""
    from src.generation import rag_pipeline
    original = rag_pipeline.urlopen
    sequence = 0

    def observed(request, **kwargs):
        nonlocal sequence
        sequence += 1
        identifier, started = sequence, perf_counter()
        entry = {"url": request.full_url, "request": json.loads(request.data)}
        response = original(request, **kwargs)

        class Response:
            def __init__(self):
                self.parts = []
            def __enter__(self):
                response.__enter__()
                return self
            def read(self, *args):
                data = response.read(*args)
                self.parts.append(data)
                return data
            def __iter__(self):
                return self
            def __next__(self):
                data = next(response)
                self.parts.append(data)
                return data
            def __getattr__(self, name):
                return getattr(response, name)
            def __exit__(self, *args):
                try:
                    return response.__exit__(*args)
                finally:
                    save(output / f"http_{identifier:03d}.json", {**entry, "seconds": perf_counter()-started,
                         "response_text": b"".join(self.parts).decode("utf-8", errors="replace")})
        return Response()
    rag_pipeline.urlopen = observed


def frozen_quality(audit, output, responses=None):
    """冻结原HTTP的题目/证据/采样参数，仅更新系统规范；不冒称检索已全部改好。"""
    from src.generation.rag_pipeline import generate_answer, _finish_generation
    base = audit / "docs/完整性测试证据/agent_rag/benchmark60"
    for index, (case, number) in enumerate((("F02", 15), ("F11", 58), ("F14", 78), ("C05", 105), ("C14", 153), ("S06", 192)), 1):
        old = json.loads((base / f"http_{number:05d}.json").read_text())
        content = old["request"]["messages"][-1]["content"]
        evidence, question = content.removeprefix("【检索上下文】\n").rsplit("\n\n【用户问题】\n", 1)
        # 位置与正文来自原真实请求，只重建引用映射，不写入期待答案。
        markers = list(re.finditer(r"\[参考文档(\d+) - 来源: (.*?)；原始块位置: (.*?)\]\n", evidence))
        references = [{"id": int(marker[1]), "source_file": marker[2], "location": marker[3], "metadata": {}, "truncated": "[正文已截断]" in evidence[marker.end():markers[i+1].start() if i+1<len(markers) else len(evidence)],
                       "text": evidence[marker.end():markers[i+1].start() if i+1<len(markers) else len(evidence)].strip().removesuffix("\n[正文已截断]").strip()}
                      for i, marker in enumerate(markers)]
        context = {"context": evidence, "references": references, "generation_mode": "grounded"}
        started = perf_counter()
        try:
            if responses:
                # 修正记录器缺truncated字段后复用本轮已收到的真实响应，不重复计费或伪造请求。
                recorded = json.loads((responses / f"http_{index:03d}.json").read_text())
                if recorded["request"]["messages"][-1]["content"] != content:
                    raise ValueError("回放的题目或上下文与冻结输入不同")
                answer = _finish_generation(json.loads(recorded["response_text"]), context, old["request"]["options"])
            else:
                answer = generate_answer(question, context, options=old["request"]["options"])
            value = {"result": answer}
        except Exception as error:
            value = {"error": repr(error)}
        save(output / f"frozen_{case}.json", {"question": question, "context_sha256": hashlib.sha256(evidence.encode()).hexdigest(),
             "baseline_http": str(base / f"http_{number:05d}.json"), "replayed_response": str(responses / f"http_{index:03d}.json") if responses else None,
             "boundary": "同一真实证据与参数的生成复测；最终Agent整链路另测。回放耗时不是模型生成耗时。", "seconds": perf_counter()-started, **value})
        print(case, "完成" if "result" in value else value, flush=True)


def agent(audit, output, workspace):
    """复用审计索引的副本运行真实Agent；不得打开原索引进行查询或写入。"""
    source = audit / ".test-tmp/completeness_retrieval/m3e_recursive512/chroma"
    index = workspace / "index"
    if not index.exists():
        shutil.copytree(source, index)
    from src.agent.react_loop import run_react
    from src.agent.tools import AVAILABLE_TOOLS
    question = "T2T-ViT 针对普通 ViT 提出的两项主要结构改进是什么？"
    started = perf_counter()
    events = list(run_react(question, AVAILABLE_TOOLS))
    save(output / "F05真实Agent.json", {"question": question, "seconds": perf_counter()-started, "events": events,
         "boundary": "真实旧审计索引副本、真实M3E/BGE/Qwen；中文查英文排名不作为本轮优化目标"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("numerical", "quality", "agent"))
    parser.add_argument("--audit", type=Path, default=Path("/Users/rean/github/测试项"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--responses", type=Path, help="仅回放同一冻结输入已记录的真实响应，不启动新的模型请求")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    workspace = ROOT / ".test-tmp/audit-repairs"
    from src.utils import config
    settings = config.load_config()
    settings["llm"]["base_url"] = "http://127.0.0.1:11439"
    settings["paths"].update(vector_index=str(workspace / "index"), raw_documents=str(args.audit / "data/raw"),
                             logs=str(args.output / "business_logs"), session_db=str(workspace / "sessions.sqlite3"))
    config.load_config = lambda: deepcopy(settings)
    import torch
    torch.set_num_threads(2)
    save(args.output / "运行配置.json", settings)
    if args.phase == "numerical":
        numerical(args.output)
    else:
        observe_http(args.output)
        if args.phase == "quality":
            frozen_quality(args.audit, args.output, args.responses)
        else:
            agent(args.audit, args.output, workspace)
