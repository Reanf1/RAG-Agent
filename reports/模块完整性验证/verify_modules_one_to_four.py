"""四模块真实联调：AppTest操作原页面，真实本地模型、Chroma、SQLite和JSONL。

仅隔离数据路径；AppTest上传不是原生浏览器文件选择或动画时序验收。
"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from streamlit.testing.v1 import AppTest
from src.agent import react_loop
from src.chunking import split_documents
from src.data_loader.pdf_loader import load_pdf
from src.frontend.components.documents import list_documents
from src.generation import rag_pipeline
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config


def serial(value):
    """保留真实Document/Message内容，不能用字符串占位丢弃正文和来源。"""
    if isinstance(value, (Document, BaseMessage)):
        return serial(value.model_dump())
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.root.exists() or args.output.exists():
        raise FileExistsError("请使用新的测试目录与报告路径，保留失败和历史证据")
    args.root.mkdir(parents=True)
    config = deepcopy(load_config())

    def business_snapshot():
        return {"config": sha256((ROOT / "config.yaml").read_bytes()).hexdigest(),
                "gitignore": sha256((ROOT / ".gitignore").read_bytes()).hexdigest(),
                "raw": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                        for p in (ROOT / load_config()["paths"]["raw_documents"]).rglob("*") if p.is_file()},
                "index_ids": sorted(d.metadata["chunk_id"] for d in VectorStore().list_chunks()),
                "session_exists": (ROOT / load_config()["paths"]["session_db"]).exists()}

    baseline = business_snapshot()
    for key, folder in (("raw_documents", "raw"), ("vector_index", "index"), ("logs", "logs")):
        config["paths"][key] = str(args.root / folder)
    config["paths"]["session_db"] = str(args.root / "memory.sqlite3")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    report = {"started_at": datetime.now().astimezone().isoformat(), "config": config,
              "baseline": baseline, "checks": {}, "quality_checks": {}, "steps": [],
              "model_packets": [], "passed": False}

    def save():
        args.output.write_text(json.dumps(serial(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def check(name, condition):
        report["checks"][name] = bool(condition)
        save()
        print(name, bool(condition), flush=True)

    real_http = react_loop.urlopen

    def record_http(request, *params, **kwargs):
        packet = {"request": json.loads(request.data)}
        report["model_packets"].append(packet)
        save()
        if packet["request"].get("stream") is True:
            # streaming模块可能在patch期间首次导入并沿用该函数；NDJSON必须原样逐行消费。
            packet["recording"] = "stream_passthrough; actual_done_and_usage_in_rag_message"
            save()
            return real_http(request, *params, **kwargs)
        try:
            with real_http(request, *params, **kwargs) as response:
                raw = response.read()
            packet["response"] = json.loads(raw)
            save()
            return BytesIO(raw)  # 只记录真实响应，不替换或修正模型文本。
        except Exception as error:
            packet["error"] = f"{type(error).__name__}: {error}"
            save()
            raise

    def page_step(name, app):
        report["steps"].append({"name": name, "exceptions": [item.message for item in app.exception],
                                "warnings": [item.value for item in app.warning],
                                "errors": [item.value for item in app.error]})
        check(name + "页面无异常", not app.exception)
        if app.exception:
            raise RuntimeError(f"{name}页面异常，详见原始报告")

    try:
        with ExitStack() as stack:
            for module in ("src.utils.config", "src.agent.memory", "src.utils.logger", "src.agent.react_loop",
                           "src.agent.tools", "src.agent.router", "src.retrieval.vector_store",
                           "src.retrieval.hybrid_retriever", "src.retrieval.reranker",
                           "src.generation.rag_pipeline", "src.generation.cache", "src.chunking"):
                stack.enter_context(patch(module + ".load_config", return_value=config))
            for module in (react_loop, rag_pipeline):
                stack.enter_context(patch.object(module, "urlopen", side_effect=record_http))
            app_path = str(ROOT / "src/frontend/app.py")
            app = AppTest.from_file(app_path, default_timeout=300).run()
            page_step("启动", app)
            source = ROOT / "data/raw/embedding_papers/attention.pdf"
            raw = source.read_bytes()
            identifier = sha256(raw).hexdigest()
            app.file_uploader[0].set_value([(source.name, raw, "application/pdf")]).run()
            app.button(key="start_import").click().run()
            page_step("真实PDF上传入库", app)
            tasks = deepcopy(app.session_state["import_tasks"])
            report["ingestion"] = [{k: t[k] for k in ("name", "status", "indexed", "chunk_count", "added_chunks")} for t in tasks]
            store = VectorStore()
            originals = {d.metadata["chunk_id"]: d for d in store.list_chunks()}
            uploaded = args.root / "raw" / identifier / source.name
            expected = {d.metadata["chunk_id"]: d for d in split_documents(load_pdf(uploaded))}
            check("PDF加载分块真实向量入库正文及元数据一致", originals == expected and bool(originals))
            check("上传原文哈希和导入进度一致", (args.root / "raw" / identifier / source.name).read_bytes() == raw
                  and app.session_state["import_progress"] == {"completed": 1, "total": 1}
                  and all(t["indexed"] and t["status"] == "success" for t in tasks))
            user, session = app.session_state["agent_user_id"], app.session_state["agent_session_id"]
            report["identity"] = {"user": user, "session": session}
            question = "已上传的 attention.pdf 论文中，编码器由多少个相同的层组成？请回答数字和原文页码。"
            app.text_input(key="agent_question").set_value(question)
            app.button(key="run_agent").click().run()
            page_step("Agent文献问答", app)
            first = deepcopy(app.session_state["agent_last_event"])
            report["agent_first"] = first
            tools = [s for s in first["metrics"]["trace"] if s["type"] == "tool_result"]
            rag = [s["result"] for s in tools if s["name"] == "knowledge_base_search" and s["status"] == "success"]
            check("Agent实际接入混合重排及RAG生成", bool(rag) and rag[0]["retrieval"]["returned_chunks"] == 5
                  and rag[0]["usage"]["prompt_eval_count"] > 0)
            check("Agent问答实际完成及完整决策轨迹", first["task_complete"] and
                  {"thought", "tool_call", "tool_result", "observation", "done"}.issubset(
                      {s["type"] for s in first["metrics"]["trace"]}))
            refs = rag[0]["citations"] if rag else []
            check("Agent引用对应入库原文与物理页码", bool(refs) and all(
                r["metadata"]["chunk_id"] in originals and all(
                    originals[r["metadata"]["chunk_id"]].metadata.get(k) == v for k, v in r["metadata"].items())
                and originals[r["metadata"]["chunk_id"]].page_content.startswith(r["text"])
                and r["source_file"] == source.name for r in refs))
            report["quality_checks"]["first_answer_has_six_and_page_three"] = bool(re.search(r"6|六", first["full_response"])) and "第3页" in first["full_response"]
            report["quality_checks"]["first_final_answer_keeps_source_filename"] = source.name in first["full_response"]
            app.text_input(key="agent_question").set_value("把你刚才回答的编码器层数乘以2，调用计算器给出算式和结果。")
            app.button(key="run_agent").click().run()
            page_step("Agent带历史追问", app)
            followup = deepcopy(app.session_state["agent_last_event"])
            report["agent_followup"] = followup
            calculations = [s for s in followup["metrics"]["trace"] if s["type"] == "tool_result" and s["name"] == "calculator"]
            check("历史层数传入实际计算器并返回12", followup["task_complete"] and bool(calculations)
                  and calculations[-1]["status"] == "success" and "12" in followup["full_response"])
            memory = app.session_state["agent_memory"]
            check("两轮Agent归属和完整历史落库", len(memory.get_messages(user, session)) == 4
                  and followup["user_id"] == user and followup["session_id"] == session)
            app.chat_input(key="rag_question").set_value(question).run()
            if "rag_pending" in app.session_state:
                report["rag_required_confirmation"] = True
                app.button(key="confirm_low_relevance").click().run()
            page_step("RAG真实流式引用", app)
            direct = deepcopy(app.session_state["rag_messages"][-1])
            report["rag_direct"] = direct
            check("RAG流式完成保留实际用量及引用", direct["complete"] and not direct.get("error")
                  and bool(direct["citations"]) and direct["usage"]["prompt_eval_count"] > 0)
            app.chat_input(key="rag_question").set_value(question).run()
            page_step("RAG精确缓存复用", app)
            cached = deepcopy(app.session_state["rag_messages"][-1])
            report["rag_cached"] = cached
            check("相同问题缓存命中且答案及引用一致", cached.get("cache", {}).get("hit") is True
                  and cached["answer"] == direct["answer"] and cached["citations"] == direct["citations"])
            app.button(key="check_health").click().run()
            report["health"] = deepcopy(app.session_state["health_result"])
            report["metrics"] = {item.label: item.value for item in app.metric}
            check("真实LLM索引健康检查及前端指标", report["health"]["status"] == "ok"
                  and report["health"]["vector_database"]["chunks"] == len(originals)
                  and report["metrics"].get("本次 Agent Token") == str(followup["metrics"]["tokens"]["total"]))
            app.button(key="new_conversation").click().run()
            check("新会话清空显示并保留旧会话", app.session_state["agent_session_id"] != session
                  and not app.session_state["agent_messages"] and not app.session_state["rag_messages"])
            app.selectbox(key="conversation_select").set_value(session).run()
            check("会话切换恢复两轮及引用快照", len(app.session_state["agent_messages"]) == 2
                  and len(app.session_state["rag_messages"]) == 2
                  and app.session_state["agent_last_event"]["request_id"] == followup["request_id"])
            packet_count = len(report["model_packets"])
            fresh = AppTest.from_file(app_path, default_timeout=300)
            fresh.query_params["visitor"], fresh.query_params["conversation"] = user, session
            fresh.run()
            page_step("地址刷新恢复", fresh)
            check("新页面刷新不调用模型且恢复相同SQLite内容", len(report["model_packets"]) == packet_count
                  and fresh.session_state["agent_messages"] == app.session_state["agent_messages"]
                  and fresh.session_state["rag_messages"] == app.session_state["rag_messages"])
            app.button(key="delete_conversation").click().run()
            app.button(key="confirm_delete_conversation").click().run()
            check("删除会话归入当前访客回收区", session not in memory.list_sessions(user)
                  and session in memory.list_sessions(user, archived=True))
            app.selectbox(key="restore_conversation_select").set_value(session).run()
            app.button(key="restore_conversation").click().run()
            check("恢复会话保留两类历史与轨迹", app.session_state["agent_session_id"] == session
                  and len(app.session_state["agent_messages"]) == 2 and len(app.session_state["rag_messages"]) == 2)
            sentinel = b"CROSS_MODULE_SENTINEL20261003: Incremental index integration fixture."
            app.file_uploader[0].set_value([("integration_fixture.md", sentinel, "text/markdown")]).run()
            app.button(key="start_import").click().run()
            page_step("新增Markdown增量入库", app)
            current = {d.metadata["chunk_id"]: d for d in store.list_chunks()}
            check("增量仅新增一块旧论文正文及ID保留", len(current) == len(originals) + 1
                  and all(current.get(key) == value for key, value in originals.items()))
            matches = BM25Retriever(store).search("CROSS_MODULE_SENTINEL20261003", k=1)
            check("BM25读取新增正文且知识库面板两份文档", bool(matches) and matches[0][0].page_content == sentinel.decode()
                  and len(list_documents(args.root / "raw", args.root / "index")) == 2)
            app.selectbox(key="manage_doc_id").set_value(identifier).run()
            app.button(key="delete_document").click().run()
            app.button(key="confirm_delete_document").click().run()
            check("删除论文移除向量且保留新增文档与历史", store.count() == 1
                  and not (args.root / "raw" / identifier).exists() and len(memory.get_messages(user, session)) == 4)
            app.selectbox(key="restore_doc_id").set_value(identifier).run()
            app.button(key="restore_document").click().run()
            page_step("论文恢复重新入库", app)
            restored = {d.metadata["chunk_id"]: d for d in store.list_chunks()}
            check("恢复重建该文档原块且保留增量块", restored == current)
            logs = {prefix: [json.loads(line) for path in (args.root / "logs").glob(prefix + "_*.jsonl")
                             for line in path.read_text().splitlines()] for prefix in ("rag", "agent")}
            report["logs"] = logs
            check("两次Agent最终日志和SQLite请求一致", {r["request_id"] for r in logs["agent"] if r["event"] == "done"}
                  == {first["request_id"], followup["request_id"]})
            report["final_library"] = list_documents(args.root / "raw", args.root / "index")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], flush=True)
    finally:
        report["business_after"] = business_snapshot()
        check("用户配置原文索引及会话库状态保留", report["business_after"] == baseline)
        report["finished_at"] = datetime.now().astimezone().isoformat()
        report["passed"] = bool(report["checks"]) and all(report["checks"].values()) and not report.get("error")
        save()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
