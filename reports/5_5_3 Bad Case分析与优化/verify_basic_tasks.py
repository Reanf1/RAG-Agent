"""真实本地模型基本流程复测；合成资料与临时索引隔离，不修改已有知识库。"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib
import json
import platform
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    from src.utils.config import load_config
    torch.set_num_threads(2)
    config = load_config()
    config["llm"]["base_url"] = args.base_url
    runtime = Path(tempfile.mkdtemp(prefix="rag-basic-tasks-"))
    for key, name in {"raw_documents": "raw", "vector_index": "index", "logs": "logs", "session_db": "memory.sqlite3"}.items():
        config["paths"][key] = str(runtime / name)
    rows = []

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
            default=lambda item: item.model_dump() if hasattr(item, "model_dump") else str(item)) + "\n", encoding="utf-8")

    def check(name, run):
        start = perf_counter()
        try:
            row = {"name": name, "status": "passed", "details": run()}
        except Exception as error:
            row = {"name": name, "status": "failed", "error": f"{type(error).__name__}: {error}"}
        row["seconds"] = round(perf_counter() - start, 3)
        rows.append(row)
        save("results.json", rows)
        print(json.dumps({key: row[key] for key in ("name", "status", "seconds")}, ensure_ascii=False), flush=True)

    save("runtime.json", {"config": config, "runtime": str(runtime), "platform": platform.platform(),
        "note": "合成文本用于验证软件基本流程，数字不是论文实验成绩；仅覆盖配置，模型、检索、工具均真实执行。",
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted((ROOT / "src").rglob("*.py"))}})
    # 仅把路径配置指向临时数据和指定的本地Ollama；不替换任何业务返回值。
    with ExitStack() as stack:
        stack.enter_context(patch("src.utils.config.load_config", return_value=config))
        for path in (ROOT / "src").rglob("*.py"):
            if path.name == "__init__.py" or path == ROOT / "src/frontend/app.py":
                continue
            module = importlib.import_module(".".join(path.relative_to(ROOT).with_suffix("").parts))
            if hasattr(module, "load_config"):
                stack.enter_context(patch.object(module, "load_config", return_value=config))
        from src.agent.memory import MemoryManager, run_session
        from src.agent.tools import get_available_tools
        from src.data_loader import create_import_tasks
        from src.generation.cache import SemanticCache
        from src.retrieval.vector_store import VectorStore, batch_build_index
        memory = MemoryManager(runtime / "memory.sqlite3")
        user = uuid4().hex
        session = memory.create_session(user)
        cache, pending = SemanticCache(), {}
        fixtures = [
            ("澄禾实验.txt", "澄禾实验：水稻叶片识别\n作者：李禾、王澄。年份：2025。\n背景：通过图像分类识别水稻叶片病害。\n方法：澄禾实验提出LeafGate分类器。\n数据集：田畴-73数据集，共160张叶片图片。\n实验结果：LeafGate测试准确率为91.6%，对照方法准确率87.2%。\n结论：在本次测试条件下，LeafGate提高识别准确率。\n声明：这是程序验收的合成资料，不是真实论文。"),
            ("雨田实验.txt", "雨田实验：水稻图像分类\n作者：张雨、刘田。年份：2025。\n背景：研究水稻叶片图像分类。\n方法：雨田实验采用DenseCrop分类器。\n数据集：田畴-73数据集，共128张图片。\n实验结果：DenseCrop测试准确率为88.4%。\n结论：该方法适用于本实验样本，未验证外部泛化。\n声明：这是程序验收的合成资料，不是真实论文。")]
        save("fixtures.json", fixtures)

        def ingest():
            tasks = create_import_tasks([(name, text.encode()) for name, text in fixtures])
            list(batch_build_index(tasks, runtime / "raw"))
            assert all(task["status"] == "success" and task["indexed"] for task in tasks), tasks
            return {"documents": len(tasks), "chunks": VectorStore().count()}

        def ask(label, question, predicate):
            registry = get_available_tools(cache=cache, session_id=session, pending=pending, request_question=question)
            events = list(run_session(question, user, session, tools=registry, memory=memory, stream=True))
            # 测试资料由本脚本创建并核对；低相关时仍走页面使用的明确确认续跑入口。
            approvals = [event for event in events if event["type"] == "tool_result"
                         and isinstance(event.get("result"), dict) and event["result"].get("confirmation_id")]
            if approvals:
                approval = pending[approvals[-1]["result"]["confirmation_id"]]
                registry = get_available_tools(cache=cache, session_id=session, confirmation=approval)
                confirmed_args = approval.get("args", {"question": approval.get("tool_question"), "doc_id": approval.get("doc_id")})
                events.extend(run_session(question, user, session, tools=registry, memory=memory,
                                          stream=True, confirmed_rag_args=confirmed_args))
            # 每个事件保留自身数据，累计账目只保存末次，避免重复复制整段轨迹。
            save(label + "-events.json", [
                {key: value for key, value in event.items()
                 if key not in {"message", "context"} and (key != "metrics" or event is events[-1])}
                for event in events])
            last = events[-1]
            results = [event for event in events if event["type"] == "tool_result"]
            assert last["type"] == "done" and predicate(last, results), {
                "answer": last.get("full_response"), "complete": last.get("task_complete"), "stop": last.get("stop_reason")}
            return {"question": question, "answer": last["full_response"], "complete": last["task_complete"],
                    "tools": [event["name"] for event in results], "confirmation_used": bool(approvals),
                    "tokens": last["metrics"]["tokens"], "stream_events": sum(event["type"] == "token" for event in events)}

        check("真实加载_M3E_Chroma", ingest)
        question = "澄禾实验.txt使用哪个数据集，测试准确率是多少？请给出来源。"
        check("完整问题_混合检索_BGE_流式引用", lambda: ask("rag", question, lambda end, results:
            end["task_complete"] and "田畴-73" in end["full_response"] and "91.6" in end["full_response"]
            and any(event["name"] == "knowledge_base_search" and event.get("result", {}).get("citations") for event in results)))
        check("文档关键词", lambda: ask("keyword", "提取澄禾实验.txt的关键词", lambda end, results:
            end["task_complete"] and any(event["name"] == "keyword_extract" and event["status"] == "success"
                and event["result"].get("doc_id") and event["result"].get("keywords") for event in results)))
        check("两文档三维对比_两组数字", lambda: ask("compare", "比较澄禾实验.txt和雨田实验.txt两篇论文的方法、数据集、实验结果。",
            lambda end, results: end["task_complete"] and all(value in end["full_response"] for value in ("LeafGate", "DenseCrop", "91.6", "88.4"))))
        check("真实计算器", lambda: ask("calculator", "计算23乘以17", lambda end, results: end["task_complete"] and "391" in end["full_response"]))
        check("多轮历史", lambda: ask("history", "刚才让我计算的算式和结果是什么？", lambda end, results:
            end["task_complete"] and all(value in end["full_response"] for value in ("23", "17", "391"))))
        # 覆盖其余本地工具；文献列表已在文件名预检中真实执行。
        check("文档元信息", lambda: ask("metadata", "查询澄禾实验.txt的论文元信息，只需标题、作者和年份。",
            lambda end, results: end["task_complete"] and all(value in end["full_response"] for value in ("李禾", "王澄", "2025"))
            and any(event["name"] == "paper_metadata" and event["status"] == "success" for event in results)))
        check("结构化摘要", lambda: ask("summary", "生成澄禾实验.txt的结构化摘要。", lambda end, results:
            end["task_complete"] and all(value in end["full_response"] for value in ("背景", "方法", "结果", "结论", "91.6"))
            and any(event["name"] == "paper_summary" and event["result"].get("citations") for event in results)))
        check("系统时间工具", lambda: ask("time", "查询当前系统时间。", lambda end, results:
            end["task_complete"] and any(event["name"] == "current_time" and event["status"] == "success"
                and abs((datetime.now(timezone.utc) - datetime.fromisoformat(event["result"]["system_time"])).total_seconds()) < 120
                for event in results)))

        def ui():
            from streamlit.testing.v1 import AppTest
            app = AppTest.from_file(str(ROOT / "src/frontend/app.py"), default_timeout=300)
            app.query_params.update(visitor=user, conversation=session)
            app.run()
            before = len(memory.get_messages(user, session))
            app.chat_input[0].set_value("计算29乘以13").run()
            assert not app.exception, [item.message for item in app.exception]
            answer = app.session_state["agent_messages"][-1]["answer"]
            assert "377" in answer and len(memory.get_messages(user, session)) == before + 2, answer
            assert any("377" in str(item.value) for item in app.markdown)
            return {"answer": answer, "mode": "AppTest真实页面调用模型并保存会话；未覆盖实机浏览器兼容性"}

        check("页面提交_保存_重绘", ui)
    return int(any(row["status"] != "passed" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
