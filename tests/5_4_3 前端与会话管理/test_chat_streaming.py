"""5.4.3 前端与会话管理：TestStreamingFrontend。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from copy import deepcopy
import json
import tempfile
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from src.utils.logger import read_rag_requests, retrieval_score_distribution
from tests.helpers import StreamingResponse


class TestStreamingFrontend(unittest.TestCase):
    """实际操作 Streamlit 聊天组件；NDJSON 样例隔离模型，不证明生成质量。"""

    def setUp(self):
        from src.utils.config import load_config
        from streamlit.testing.v1 import AppTest
        self.config = load_config()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config["paths"]["raw_documents"] = str(Path(self.directory.name) / "raw")
        self.config["paths"]["session_db"] = str(Path(self.directory.name) / "memory.sqlite3")
        self.config["paths"]["vector_index"] = str(Path(self.directory.name) / "index")
        self.config["paths"]["logs"] = str(Path(self.directory.name) / "logs")
        for target in ("src.utils.config.load_config", "src.generation.rag_pipeline.load_config",
                       "src.generation.cache.load_config", "src.utils.logger.load_config"):
            patcher = patch(target, return_value=self.config)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("src.retrieval.hybrid_retriever.HybridRetriever")
        self.retriever = patcher.start().return_value
        self.addCleanup(patcher.stop)
        patcher = patch("src.generation.cache.get_embeddings")
        self.cache_embedding = patcher.start().return_value
        self.cache_embedding.embed_query.return_value = [1.0, 0.0]
        self.addCleanup(patcher.stop)
        self.retriever.search.return_value = [(Document(page_content="编码器有6层。", metadata={
            "source_file": "attention.pdf", "page_number": 3, "chunk_id": "功能样例块"}), 0.8)]
        patcher = patch("src.generation.streaming.urlopen")
        self.opener = patcher.start()
        self.addCleanup(patcher.stop)
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "src/frontend/app.py"),
                                    default_timeout=10).run()

    def reply(self, *, done=True):
        packets = [{"message": {"content": chunk}, "done": False}
                   for chunk in ("编码器有6层。", "[参", "考文档1", "]")]
        if done:
            packets.append({"model": "qwen2.5:7b", "done": True, "message": {"content": ""},
                            "done_reason": "stop", "prompt_eval_count": 400, "eval_count": 20})
        self.opener.return_value = StreamingResponse(packets)

    def test_page_start_is_lazy_and_chat_shows_sources_and_actual_usage(self):
        self.retriever.search.assert_not_called()
        self.opener.assert_not_called()
        self.reply()
        self.app.chat_input(key="rag_question").set_value("层数？").run()
        self.assertFalse(self.app.exception)
        self.retriever.search.assert_called_once_with("层数？", k=5, rerank=True)
        message = self.app.session_state["rag_messages"][0]
        self.assertTrue(message["complete"])
        self.assertIn("attention.pdf；第3页", message["answer"])
        self.assertEqual(self.app.text[0].value, "编码器有6层。")
        self.assertTrue(any("输入 Token 400 · 输出 Token 20" in item.value for item in self.app.caption))

    def test_history_rerun_does_not_generate_again_and_clear_removes_it(self):
        self.reply()
        self.app.chat_input[0].set_value("层数？").run()
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.opener.call_count, 1)
        self.assertEqual(len(self.app.chat_message), 2)
        self.app.button(key="clear_rag_chat").click().run()
        self.assertEqual(self.app.session_state["rag_messages"], [])
        self.assertEqual(len(self.app.chat_message), 0)

    def test_incomplete_stream_is_visible_error_with_partial_answer(self):
        self.reply(done=False)
        self.app.chat_input[0].set_value("层数？").run()
        self.assertFalse(self.app.exception)
        message = self.app.session_state["rag_messages"][0]
        self.assertFalse(message["complete"])
        self.assertIn("未收到完成标记", message["error"])
        self.assertIn("编码器有6层", message["answer"])
        self.assertTrue(any("回答未完成" in item.value for item in self.app.error))
        self.assertFalse(any("服务已结束" in item.value for item in self.app.caption))

    def test_empty_library_notice_and_retrieval_failure_never_hide_errors(self):
        self.retriever.search.return_value = []
        self.reply()
        self.app.chat_input[0].set_value("层数？").run()
        self.assertTrue(any("以下回答没有文献依据" in item.value for item in self.app.info))
        self.assertIn("无效引用", self.app.session_state["rag_messages"][0]["answer"])
        self.retriever.search.side_effect = RuntimeError("本地模型不可用")
        self.app.chat_input[0].set_value("另一个问题").run()
        self.assertFalse(self.app.exception)
        self.assertIn("本地模型不可用", self.app.session_state["rag_messages"][-1]["error"])
        self.assertEqual(self.opener.call_count, 1)

    def test_repeated_and_similar_questions_skip_retrieval_and_model_and_keep_sources(self):
        self.reply()
        self.app.chat_input[0].set_value("层数？").run()
        first = self.app.session_state["rag_messages"][0]
        self.cache_embedding.embed_query.reset_mock()
        self.app.chat_input[0].set_value("层数？").run()
        self.cache_embedding.embed_query.assert_not_called()
        self.app.chat_input[0].set_value("编码器有几层？").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.opener.call_count, 1)
        self.assertEqual(self.retriever.search.call_count, 1)
        messages = self.app.session_state["rag_messages"]
        self.assertEqual(messages[1]["cache"]["mode"], "exact")
        self.assertEqual(messages[2]["cache"]["mode"], "semantic")
        self.assertEqual(messages[2]["citations"], first["citations"])
        self.assertEqual(messages[2]["usage"]["eval_count"], 0)
        self.assertTrue(any("缓存已返回" in item.value for item in self.app.caption))

    def test_knowledge_change_and_clear_invalidate_cache(self):
        self.reply()
        self.app.chat_input[0].set_value("层数？").run()
        self.retriever.vector_store.list_chunks.return_value = [Document(page_content="新文献")]
        self.reply()
        self.app.chat_input[0].set_value("层数？").run()
        self.assertEqual(self.opener.call_count, 2)
        self.app.button(key="clear_rag_chat").click().run()
        self.assertFalse(self.app.session_state["rag_cache"].entries)
        self.reply()
        self.app.chat_input[0].set_value("层数？").run()
        self.assertEqual(self.opener.call_count, 3)

    def test_interrupted_answer_is_not_cached_and_retry_calls_model(self):
        self.reply(done=False)
        self.app.chat_input[0].set_value("层数？").run()
        self.assertFalse(self.app.session_state["rag_cache"].entries)
        self.reply()
        self.app.chat_input[0].set_value("层数？").run()
        self.assertEqual(self.opener.call_count, 2)
        self.assertTrue(self.app.session_state["rag_messages"][-1]["complete"])

    def test_changed_generation_settings_do_not_reuse_old_answer(self):
        self.reply()
        self.app.chat_input[0].set_value("层数？").run()
        self.config["llm"]["temperature"] = 0.2
        self.reply()
        self.app.chat_input[0].set_value("层数？").run()
        self.assertEqual(self.opener.call_count, 2)
        self.assertEqual(json.loads(self.opener.call_args.args[0].data)["options"]["temperature"], 0.2)

    def test_cache_write_failure_keeps_completed_answer_and_gives_notice(self):
        self.cache_embedding.embed_query.side_effect = RuntimeError("缓存向量化失败")
        self.reply()
        self.app.chat_input[0].set_value("层数？").run()
        message = self.app.session_state["rag_messages"][-1]
        self.assertTrue(message["complete"])
        self.assertNotIn("error", message)
        self.assertTrue(any("缓存未写入" in item.value for item in self.app.warning))

    def test_knowledge_changed_during_generation_does_not_store_stale_answer(self):
        self.retriever.vector_store.list_chunks.side_effect = [[], [Document(page_content="新加入")]]
        self.reply()
        self.app.chat_input[0].set_value("层数？").run()
        self.assertTrue(self.app.session_state["rag_messages"][-1]["complete"])
        self.assertFalse(self.app.session_state["rag_cache"].entries)


    def test_low_relevance_waits_for_confirmation_then_generates_once(self):
        document = self.retriever.search.return_value[0][0]
        self.retriever.search.return_value = [(document, 0.02)]
        self.app.chat_input[0].set_value("层数？").run()
        self.assertFalse(self.app.exception)
        self.opener.assert_not_called()
        self.assertEqual(self.app.session_state["rag_messages"], [])
        self.assertIn("编码器有6层", self.app.text[0].value)
        self.assertTrue(any("相关性低" in item.value for item in self.app.warning))
        self.app.run()
        self.opener.assert_not_called()
        self.reply()
        self.app.button(key="confirm_low_relevance").click().run()
        self.assertFalse(self.app.exception)
        message = self.app.session_state["rag_messages"][-1]
        self.assertTrue(message["complete"])
        self.assertEqual(message["generation_mode"], "low")
        self.assertIn("相关性低", message["answer"])
        self.assertEqual(message["citations"][0]["source_file"], "attention.pdf")
        self.assertFalse(self.app.session_state["rag_cache"].entries)
        self.app.run()
        self.assertEqual(self.opener.call_count, 1)
        self.app.chat_input[0].set_value("层数？").run()
        self.assertIn("rag_pending", self.app.session_state)
        self.assertEqual(self.opener.call_count, 1)

    def test_cancel_and_clear_remove_pending_request_without_generation(self):
        self.retriever.search.return_value = [(self.retriever.search.return_value[0][0], 0.01)]
        self.app.chat_input[0].set_value("层数？").run()
        self.app.button(key="cancel_low_relevance").click().run()
        self.assertNotIn("rag_pending", self.app.session_state)
        self.assertEqual(self.app.session_state["rag_messages"], [])
        self.app.chat_input[0].set_value("层数？").run()
        self.app.button(key="clear_rag_chat").click().run()
        self.assertNotIn("rag_pending", self.app.session_state)
        self.opener.assert_not_called()

    def test_changed_knowledge_or_configuration_cannot_confirm_stale_context(self):
        self.retriever.search.return_value = [(self.retriever.search.return_value[0][0], 0.01)]
        for change in ("knowledge", "config"):
            self.app.chat_input[0].set_value("层数？").run()
            if change == "knowledge":
                self.retriever.vector_store.list_chunks.return_value = [Document(page_content="新文献")]
            else:
                self.config["generation"]["low_relevance_threshold"] = 0.2
            self.app.button(key="confirm_low_relevance").click().run()
            self.assertFalse(self.app.exception)
            message = self.app.session_state["rag_messages"][-1]
            self.assertFalse(message["complete"])
            self.assertIn("已改变", message["error"])
        self.opener.assert_not_called()

    def test_new_question_replaces_pending_low_relevance_question(self):
        document = self.retriever.search.return_value[0][0]
        self.retriever.search.return_value = [(document, 0.01)]
        self.app.chat_input[0].set_value("旧问题").run()
        self.retriever.search.return_value = [(document, 0.8)]
        self.reply()
        self.app.chat_input[0].set_value("新问题").run()
        self.assertNotIn("rag_pending", self.app.session_state)
        self.assertEqual(self.app.session_state["rag_messages"][0]["question"], "新问题")
        self.assertEqual(self.opener.call_count, 1)

    def test_empty_notice_and_error_advice_survive_history_rerun(self):
        self.retriever.search.return_value = []
        self.reply()
        self.app.chat_input[0].set_value("概念？").run()
        self.app.run()
        self.assertTrue(any("纯模型回答" in item.value for item in self.app.info))
        self.assertTrue(self.app.session_state["rag_messages"][0]["answer"].startswith("当前知识库中未找到相关文档。"))
        self.opener.side_effect = TimeoutError("mock timeout")
        self.app.chat_input[0].set_value("重试问题").run()
        self.app.run()
        message = self.app.session_state["rag_messages"][-1]
        self.assertFalse(message["complete"])
        self.assertIn("超时", message["error"])
        self.assertTrue(any("缩短问题" in item.value for item in self.app.info))
        self.assertFalse(self.app.session_state["rag_cache"].entries)
        self.assertEqual(self.opener.call_count, 2)


    def test_request_log_keeps_full_topk_separate_from_truncated_context(self):
        text = "长原文。" * 1800
        first = Document(page_content=text, metadata={"source_file": "长论文.pdf", "page_number": 5, "chunk_id": "long"})
        second = Document(page_content="其他候选原文。", metadata={"source_file": "论文B.pdf", "page_number": 2, "chunk_id": "other"})
        self.retriever.search.return_value = [(first, 0.8), (second, 0.2)]
        self.reply()
        self.app.chat_input[0].set_value("层数？").run()
        records, invalid = read_rag_requests()
        self.assertEqual((len(records), invalid), (1, 0))
        record = records[0]
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["retrieval"]["documents"][0]["text"], text)
        self.assertEqual(len(record["retrieval"]["documents"]), 2)
        self.assertTrue(record["context"]["truncated"])
        self.assertLess(len(record["context"]["references"][0]["text"]), len(text))
        self.assertEqual(record["retrieval"]["top1_score"], 0.8)
        self.assertEqual(record["tokens"]["total"], 420)
        self.assertEqual(record["answer"], self.app.session_state["rag_messages"][0]["answer"])
        self.assertGreater(record["timing"]["generation_seconds"], 0)
        self.assertTrue(any("样本 1" in item.value for item in self.app.caption))
        self.assertEqual(len(self.app.get("vega_lite_chart")), 1)
        self.app.run()
        self.assertEqual(len(read_rag_requests()[0]), 1)

    def test_cache_hit_logs_zero_current_tokens_and_no_new_retrieval_score(self):
        self.reply()
        self.app.chat_input[0].set_value("层数？").run()
        self.app.chat_input[0].set_value("层数？").run()
        records, invalid = read_rag_requests()
        self.assertEqual((len(records), invalid), (2, 0))
        cached = next(row for row in records if row["cache"]["hit"])
        self.assertEqual(cached["tokens"]["total"], 0)
        self.assertEqual(cached["original_usage"]["eval_count"], 20)
        self.assertEqual(cached["retrieval"]["status"], "skipped_cache")
        self.assertIsNone(cached["retrieval"]["top1_score"])
        self.assertEqual(cached["retrieval"]["documents"], [])
        self.assertEqual(cached["citations"][0]["source_file"], "attention.pdf")
        summary = retrieval_score_distribution()
        self.assertEqual(summary["cache_hits"], 1)
        self.assertEqual(summary["distributions"][0]["count"], 1)

    def test_pending_confirmation_updates_one_log_and_one_score_sample(self):
        self.retriever.search.return_value = [(self.retriever.search.return_value[0][0], 0.01)]
        self.app.chat_input[0].set_value("层数？").run()
        before = read_rag_requests()[0][0]
        self.assertEqual(before["status"], "awaiting_confirmation")
        self.assertEqual(before["tokens"]["total"], 0)
        self.reply()
        self.app.button(key="confirm_low_relevance").click().run()
        records, _ = read_rag_requests()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["request_id"], before["request_id"])
        self.assertEqual(records[0]["status"], "completed")
        self.assertTrue(records[0]["context"]["confirmed"])
        self.assertEqual(records[0]["tokens"]["total"], 420)
        self.assertEqual(retrieval_score_distribution()["distributions"][0]["count"], 1)

    def test_cancel_supersede_and_clear_keep_request_records(self):
        self.retriever.search.return_value = [(self.retriever.search.return_value[0][0], 0.01)]
        self.app.chat_input[0].set_value("旧问题").run()
        self.app.chat_input[0].set_value("新问题").run()
        self.app.button(key="cancel_low_relevance").click().run()
        self.app.chat_input[0].set_value("清空前问题").run()
        self.app.button(key="clear_rag_chat").click().run()
        records, _ = read_rag_requests()
        self.assertEqual(len(records), 3)
        self.assertEqual([row["status"] for row in records], ["superseded", "cancelled", "cancelled"])
        self.assertTrue(all(row["tokens"]["total"] == 0 for row in records))
        self.assertEqual(retrieval_score_distribution()["distributions"][0]["count"], 3)
        self.opener.assert_not_called()

    def test_partial_failure_and_retrieval_failure_log_different_token_states(self):
        self.reply(done=False)
        self.app.chat_input[0].set_value("部分回答").run()
        record = read_rag_requests()[0][0]
        self.assertEqual(record["status"], "error")
        self.assertIn("编码器有6层", record["raw_answer"])
        self.assertIn("未收到完成标记", record["error"])
        self.assertEqual(record["tokens"]["source"], "unavailable")
        self.assertIsNone(record["tokens"]["total"])
        self.retriever.search.side_effect = RuntimeError("检索失败样例")
        self.app.chat_input[0].set_value("检索失败").run()
        latest = read_rag_requests()[0][-1]
        self.assertEqual(latest["retrieval"]["status"], "error")
        self.assertEqual(latest["tokens"]["total"], 0)
        self.assertGreater(latest["timing"]["retrieval_seconds"], 0)
        self.assertEqual(retrieval_score_distribution()["failed_retrievals"], 1)
        self.assertEqual(self.opener.call_count, 1)

    def test_logging_failure_warns_without_losing_completed_answer(self):
        self.reply()
        with patch("src.utils.logger.record_rag_request", side_effect=OSError("日志目录不可写")):
            self.app.chat_input[0].set_value("层数？").run()
        self.assertFalse(self.app.exception)
        message = self.app.session_state["rag_messages"][0]
        self.assertTrue(message["complete"])
        self.assertNotIn("error", message)
        self.assertTrue(any("请求日志未保存" in item.value for item in self.app.warning))
        self.assertTrue(self.app.session_state["rag_cache"].entries)


    def test_sessions_restore_rag_citations_after_switch_and_refresh(self):
        from streamlit.testing.v1 import AppTest
        self.reply()
        app = self.app
        a = app.session_state["rag_session_id"]
        app.chat_input(key="rag_question").set_value("会话A层数？").run()
        self.assertIn("会话A层数？", app.selectbox(key="conversation_select").options[0])
        stored = deepcopy(app.session_state["rag_messages"][0])
        app.button(key="new_conversation").click().run()
        self.assertEqual(app.session_state["rag_messages"], [])
        self.assertFalse(app.session_state["rag_cache"].entries)
        self.reply()
        app.chat_input(key="rag_question").set_value("会话B层数？").run()
        app.selectbox(key="conversation_select").set_value(a).run()
        self.assertEqual(app.session_state["rag_messages"], [stored])
        self.assertEqual(len(app.chat_message), 2)
        refreshed = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "src/frontend/app.py"), default_timeout=10)
        refreshed.query_params.update(app.query_params)
        refreshed.run()
        self.assertFalse(refreshed.exception)
        self.assertEqual(refreshed.session_state["rag_messages"], [stored])
        self.assertEqual(refreshed.text[0].value, "编码器有6层。")
        self.assertEqual(self.opener.call_count, 2)
        refreshed.button(key="clear_rag_chat").click().run()
        self.assertEqual(refreshed.session_state["agent_memory"].get_rag_messages(refreshed.session_state["agent_user_id"], a), [])

    def test_switch_cancels_low_relevance_request_without_generating(self):
        self.retriever.search.return_value[0] = (self.retriever.search.return_value[0][0], 0.05)
        self.app.chat_input(key="rag_question").set_value("低相关问题").run()
        self.assertIn("rag_pending", self.app.session_state)
        a = self.app.session_state["rag_session_id"]
        self.app.button(key="new_conversation").click().run()
        self.assertFalse(self.app.exception)
        self.assertNotIn("rag_pending", self.app.session_state)
        self.assertEqual(self.app.session_state["rag_messages"], [])
        self.app.selectbox(key="conversation_select").set_value(a).run()
        self.assertNotIn("rag_pending", self.app.session_state)
        records, _ = read_rag_requests()
        self.assertEqual(records[-1]["status"], "cancelled")
        self.opener.assert_not_called()

    def test_history_write_failure_is_visible_and_partial_text_is_kept(self):
        self.reply(done=False)
        with patch("src.agent.memory.MemoryManager.append_rag_message", side_effect=OSError("磁盘已满")):
            self.app.chat_input(key="rag_question").set_value("保存失败问题").run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("磁盘已满" in w.value for w in self.app.warning))
        self.assertTrue(self.app.session_state["rag_messages"][0]["answer"])
        self.assertFalse(self.app.session_state["rag_messages"][0]["complete"])
        self.assertEqual(self.app.session_state["agent_memory"].get_rag_messages(
            self.app.session_state["agent_user_id"], self.app.session_state["rag_session_id"]), [])


if __name__ == "__main__":
    unittest.main()
