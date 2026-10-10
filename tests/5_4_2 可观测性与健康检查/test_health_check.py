"""5.4.2 可观测性与健康检查：TestHealthCheck、TestHealthCheckPage。"""

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
from src.retrieval.vector_store import VectorStore
from tests.helpers import SmallEmbeddings


class TestHealthCheck(unittest.TestCase):
    """隔离Ollama HTTP，Chroma使用真实临时数据库；不生成回答或编码向量。"""

    def setUp(self):
        import httpx
        from src.utils.config import load_config
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = load_config()
        self.config["paths"]["vector_index"] = self.directory.name
        self.config["paths"]["logs"] = str(Path(self.directory.name) / "logs")
        self.embeddings = SmallEmbeddings()
        self.store = VectorStore(self.directory.name, self.embeddings)
        for name in ("src.utils.config.load_config", "src.retrieval.vector_store.load_config"):
            patcher = patch(name, return_value=self.config)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("src.utils.config.httpx.Client")
        self.http = patcher.start()
        self.addCleanup(patcher.stop)
        self.get = self.http.return_value.__enter__.return_value.get
        self.response = httpx.Response(200, json={"models": [{"name": self.config["llm"]["model"]}]},
                                       request=httpx.Request("GET", "http://localhost:11434/api/tags"))
        self.get.return_value = self.response

    def check(self):
        from src.utils.config import check_health
        return check_health()

    def test_existing_empty_database_is_available_not_missing(self):
        result = self.check()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["vector_database"]["chunks"], 0)
        self.assertTrue(result["llm"]["service_reachable"])
        self.assertTrue(result["llm"]["model_available"])
        json.dumps(result, allow_nan=False)

    def test_nonempty_database_keeps_chunks_metadata_and_no_embedding_calls(self):
        chunks = [Document(page_content="实际临时论文块", metadata={"doc_id": "paper", "chunk_id": "chunk"})]
        self.store.add_chunks(chunks)
        calls = list(self.embeddings.document_calls)
        metadata = deepcopy(self.store._store._collection.metadata)
        with patch("src.retrieval.vector_store.get_embeddings") as model:
            result = self.check()
        model.assert_not_called()
        self.assertEqual(result["vector_database"]["chunks"], 1)
        self.assertEqual(self.store.list_chunks(), chunks)
        self.assertEqual(self.store._store._collection.metadata, metadata)
        self.assertEqual(self.embeddings.document_calls, calls)
        self.assertEqual(self.embeddings.query_calls, [])

    def test_connection_failure_does_not_skip_database_check(self):
        import httpx
        self.get.side_effect = httpx.ConnectError("服务未启动")
        result = self.check()
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["llm"]["status"], "error")
        self.assertFalse(result["llm"]["service_reachable"])
        self.assertIsNone(result["llm"]["model_available"])
        self.assertEqual(result["vector_database"]["status"], "ok")
        self.assertIn("服务未启动", result["llm"]["detail"])

    def test_health_uses_same_index_alias_as_business_client(self):
        """不能在中文原路径另开一个Chroma客户端；英文入口指向同一索引。"""
        alias = Path(self.directory.name) / "alias"
        alias.symlink_to(self.directory.name, target_is_directory=True)
        with patch("src.utils.chroma_path.chroma_persist_path", return_value=str(alias)) as path:
            result = self.check()
        path.assert_called_once_with(Path(self.directory.name))
        self.assertEqual(result["vector_database"]["status"], "ok")
        self.assertEqual(result["vector_database"]["chunks"], 0)

    def test_network_timeout_has_friendly_error_and_three_second_limit(self):
        import httpx
        self.get.side_effect = httpx.ReadTimeout("响应超时")
        result = self.check()
        self.assertIn("响应超时", result["llm"]["detail"])
        self.http.assert_called_once_with(timeout=3, trust_env=False, follow_redirects=False)
        self.get.assert_called_once_with(self.config["llm"]["base_url"].rstrip("/") + "/api/tags")
        self.assertGreaterEqual(result["llm"]["seconds"], 0)

    def test_reachable_service_missing_configured_model_is_not_ready(self):
        import httpx
        self.get.return_value = httpx.Response(200, json={"models": []}, request=self.response.request)
        result = self.check()
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["llm"]["status"], "model_missing")
        self.assertTrue(result["llm"]["service_reachable"])
        self.assertFalse(result["llm"]["model_available"])

    def test_model_without_tag_matches_latest_only(self):
        import httpx
        self.config["llm"]["model"] = "qwen2.5"
        for installed, expected in (("qwen2.5:latest", True), ("qwen2.5:7b", False)):
            with self.subTest(installed=installed):
                self.get.return_value = httpx.Response(200, json={"models": [{"name": installed}]}, request=self.response.request)
                self.assertEqual(self.check()["llm"]["model_available"], expected)

    def test_cloud_or_wrong_provider_is_rejected_without_network(self):
        for provider, address in (("ollama", "https://example.com"), ("openai", "http://localhost:11434")):
            with self.subTest(provider=provider):
                self.config["llm"].update(provider=provider, base_url=address)
                self.assertEqual(self.check()["llm"]["status"], "error")
        self.http.assert_not_called()

    def test_http_error_and_redirect_do_not_report_reachable_service(self):
        import httpx
        for status in (302, 500):
            with self.subTest(status=status):
                self.get.return_value = httpx.Response(status, request=self.response.request)
                result = self.check()
                self.assertEqual(result["llm"]["status"], "error")
                self.assertFalse(result["llm"]["service_reachable"])

    def test_invalid_json_or_model_list_is_error_not_model_missing(self):
        import httpx
        for body in (b"bad json", b'{}', b'{"models":null}', b'{"models":[{}]}', b'{"models":[42]}'):
            with self.subTest(body=body):
                self.get.return_value = httpx.Response(200, content=body, request=self.response.request)
                result = self.check()
                self.assertEqual(result["llm"]["status"], "error")
                self.assertTrue(result["llm"]["service_reachable"])
                self.assertIsNone(result["llm"]["model_available"])

    def test_missing_index_never_creates_directory_or_database(self):
        path = Path(self.directory.name) / "not-created"
        self.config["paths"]["vector_index"] = str(path)
        result = self.check()
        self.assertEqual(result["vector_database"]["status"], "not_initialized")
        self.assertIsNone(result["vector_database"]["chunks"])
        self.assertFalse(path.exists())

    def test_missing_collection_never_creates_a_new_collection(self):
        self.config["retrieval"]["collection_name"] = "missing_collection"
        result = self.check()
        self.assertEqual(result["vector_database"]["status"], "error")
        self.assertEqual([c.name for c in self.store._store._client.list_collections()], ["paper_chunks"])

    def test_index_version_or_parameters_mismatch_is_not_available(self):
        for section, key, value in (("embedding", "revision", "different"), ("retrieval", "search_ef", 101)):
            old = self.config[section][key]
            with self.subTest(key=key):
                self.config[section][key] = value
                result = self.check()
                self.assertEqual(result["vector_database"]["status"], "error")
                self.assertIn("不一致", result["vector_database"]["detail"])
            self.config[section][key] = old

    def test_database_heartbeat_failure_keeps_independent_llm_status(self):
        with patch("chromadb.api.client.Client.heartbeat", side_effect=RuntimeError("数据库连接失败")):
            result = self.check()
        self.assertEqual(result["llm"]["status"], "ok")
        self.assertEqual(result["vector_database"]["status"], "error")
        self.assertIn("数据库连接失败", result["vector_database"]["detail"])

    def test_corrupt_sqlite_file_is_reported_as_database_error(self):
        path = Path(self.directory.name) / "corrupt"
        path.mkdir()
        (path / "chroma.sqlite3").write_bytes(b"corrupt sqlite")
        self.config["paths"]["vector_index"] = str(path)
        result = self.check()
        self.assertEqual(result["vector_database"]["status"], "error")
        self.assertIsNone(result["vector_database"]["chunks"])

    def test_invalid_directory_or_backend_is_reported_as_database_error(self):
        path = Path(self.directory.name) / "file"
        path.write_text("这不是索引目录")
        self.config["paths"]["vector_index"] = str(path)
        self.assertIn("不是目录", self.check()["vector_database"]["detail"])
        self.config["retrieval"]["vector_store"] = "faiss"
        self.assertIn("Chroma", self.check()["vector_database"]["detail"])


class TestHealthCheckPage(unittest.TestCase):
    """页面按需调用，实际检查函数另有真实数据库测试。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        from src.utils.config import load_config
        config = load_config()
        config["paths"]["session_db"] = str(Path(self.directory.name) / "memory.sqlite3")
        config_patcher = patch("src.utils.config.load_config", return_value=config)
        config_patcher.start()
        self.addCleanup(config_patcher.stop)
        self.result = {"status": "ok", "checked_at": "2026-10-03T12:00:00+08:00",
                       "llm": {"status": "ok", "detail": "模型服务正常，未执行推理。", "seconds": 0.01},
                       "vector_database": {"status": "ok", "detail": "集合可读取。", "seconds": 0.02,
                                           "chunks": 0, "collection": "paper_chunks"}}
        patcher = patch("src.utils.config.check_health", side_effect=lambda: deepcopy(self.result))
        self.checker = patcher.start()
        self.addCleanup(patcher.stop)

    def page(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(Path(__file__).resolve().parents[2] / "src/frontend/app.py"), default_timeout=10).run()

    def test_page_startup_shows_sidebar_status_and_rerun_reuses_snapshot(self):
        app = self.page()
        self.assertFalse(app.exception)
        self.checker.assert_called_once()
        self.assertEqual(app.sidebar.header[0].value, "系统状态")
        self.assertTrue(app.sidebar.caption[0].value.startswith("系统时间："))
        self.assertEqual([row.value for row in app.sidebar.success],
                         ["LLM服务：可连接，模型已安装（qwen2.5:7b）", "向量数据库：索引可读取（Chroma）"])
        self.assertTrue(any("本页尚无实际推理记录" in row.value for row in app.sidebar.caption))
        self.assertFalse(any("系统健康检查" in row.value for row in app.markdown))
        app.run()
        self.checker.assert_called_once()

    def test_manual_check_shows_both_results_and_rerun_does_not_probe_again(self):
        app = self.page()
        app.button(key="check_health").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.success), 2)
        app.run()
        self.assertEqual(self.checker.call_count, 2)
        app.button(key="check_health").click().run()
        self.assertEqual(self.checker.call_count, 3)

    def test_business_success_is_retained_on_health_refresh_but_not_on_index_change(self):
        app = self.page()
        app.session_state["runtime_checks"] = {"llm": "2026-10-09T12:00:00+08:00", "vector_database": "2026-10-09T12:00:01+08:00"}
        app.button(key="check_health").click().run()
        self.assertTrue(any("最近实际推理成功：2026-10-09" in row.value for row in app.sidebar.caption))
        del app.session_state["health_result"]
        app.run()
        self.assertTrue(any("本页尚无实际推理记录" in row.value for row in app.sidebar.caption))

    def test_runtime_records_exclude_cache_failed_calls_and_bm25_only(self):
        from src.frontend.components.trace import record_runtime_success
        checks = {}
        result = {"model": "qwen2.5:7b", "usage": {"eval_count": 20}, "retrieval": {"status": "success"}}
        event = {"type": "tool_result", "name": "knowledge_base_search", "status": "success", "result": result}
        record_runtime_success(checks, {**event, "result": {**result, "cache": {"hit": True}}}, "qwen2.5:7b", "缓存")
        record_runtime_success(checks, {**event, "status": "error"}, "qwen2.5:7b", "失败")
        self.assertEqual(checks, {})
        record_runtime_success(checks, event, "qwen2.5:7b", "实际响应")
        self.assertEqual(checks, {"llm": "实际响应", "vector_database": "实际响应"})
        record_runtime_success(checks, {"type": "observation", "model": "other", "usage": {"eval_count": 30}}, "qwen2.5:7b", "其他模型")
        self.assertEqual(checks["llm"], "实际响应")

    def test_model_change_refreshes_connection_snapshot_and_clears_old_inference(self):
        app = self.page()
        app.session_state["runtime_checks"] = {"llm": "旧模型时间"}
        app.session_state["runtime_owner"] = ("http://localhost:11434", "old-model", "old-index", "old-collection")
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(self.checker.call_count, 2)
        self.assertEqual(app.session_state["runtime_checks"], {})

    def test_failed_llm_and_missing_index_are_visible_independently(self):
        self.result["status"] = "degraded"
        self.result["llm"].update(status="error", detail="连接失败，请启动Ollama。")
        self.result["vector_database"].update(status="not_initialized", detail="请先导入文档。", chunks=None)
        app = self.page()
        app.button(key="check_health").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("启动Ollama" in c.value for c in app.error))
        self.assertTrue(any("导入文档" in c.value for c in app.warning))

    def test_missing_model_and_database_failure_are_not_green(self):
        self.result["status"] = "degraded"
        self.result["llm"].update(status="model_missing", detail="配置模型未安装。")
        self.result["vector_database"].update(status="error", detail="索引损坏。", chunks=None)
        app = self.page()
        app.button(key="check_health").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.success)
        self.assertTrue(any("未安装" in c.value for c in app.warning))
        self.assertTrue(any("索引损坏" in c.value for c in app.error))


if __name__ == "__main__":
    unittest.main()
