"""5.4.4 端到端联调与测试：TestImportFrontend。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tempfile
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from src.data_loader import batch_import, create_import_tasks
from src.retrieval.vector_store import VectorStore
from tests.helpers import SmallEmbeddings


class TestImportFrontend(unittest.TestCase):
    """真实上传组件与 Chroma，隔离原文/索引；小型向量隔离大模型。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        from src.utils.config import load_config
        config = load_config()
        config["paths"]["raw_documents"] = str(Path(self.directory.name) / "raw")
        config["paths"]["session_db"] = str(Path(self.directory.name) / "memory.sqlite3")
        config["paths"]["vector_index"] = str(Path(self.directory.name) / "index")
        config["paths"]["logs"] = str(Path(self.directory.name) / "logs")
        self.embeddings = SmallEmbeddings()
        for target, value in (("src.utils.config.load_config", config),
                              ("src.utils.config.check_health", {"llm": {"status": "ok"},
                                                               "vector_database": {"status": "ok"}}),
                              ("src.utils.logger.load_config", config),
                              ("src.retrieval.vector_store.load_config", config),
                              ("src.retrieval.hybrid_retriever.load_config", config),
                              ("src.retrieval.reranker.load_config", config),
                              ("src.retrieval.vector_store.get_embeddings", self.embeddings)):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        from streamlit.testing.v1 import AppTest
        app_path = Path(__file__).resolve().parents[2] / "src/frontend/app.py"
        # 新环境首次加载界面依赖较慢，避免默认 3 秒等待导致误报。
        self.app = AppTest.from_file(str(app_path), default_timeout=10).run()

    def delete_button(self, app=None, doc_id=None):
        """根据真实文档ID找到行内删除按钮，支持页面刷新后无上传任务的情况。"""
        from src.frontend.components.documents import list_documents
        if doc_id is None:
            doc_id = list_documents(Path(self.directory.name) / "raw", Path(self.directory.name) / "index")[0]["doc_id"]
        return (app or self.app).button(key=f"delete_document:{doc_id}")


    def test_batch_upload_progress_state_and_rerun(self):
        """实际操作上传/按钮，核验混合结果、完整进度和重跑不重复执行。"""
        app = self.app
        self.assertTrue(app.button(key="start_import").disabled)
        # 开发过程中页面可能保留上一阶段“只加载成功”的任务，不得显示索引成功。
        loaded = create_import_tasks([("good.txt", "中文正文".encode("utf-8"))])
        list(batch_import(loaded, Path(self.directory.name) / "raw"))
        app.session_state["import_tasks"] = loaded
        app.file_uploader[0].set_value([("good.txt", "中文正文".encode("utf-8"), "text/plain")]).run()
        self.assertFalse(app.sidebar.dataframe)
        self.assertFalse(app.get("progress"))
        self.assertFalse(app.button(key="start_import").disabled)
        app.file_uploader[0].set_value([
            ("good.txt", "中文正文".encode("utf-8"), "text/plain"),
            ("bad.txt", b"\xff", "text/plain"),
        ]).run()
        app.button(key="start_import").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(list(app.sidebar.dataframe[0].value["状态"]), ["成功", "失败"])
        self.assertEqual(list(app.sidebar.dataframe[0].value.columns), ["文件", "状态"])
        self.assertFalse(app.file_uploader[0].value)
        self.assertTrue(app.session_state["import_tasks"][0]["indexed"])
        self.assertEqual(app.session_state["import_progress"], {"completed": 2, "total": 2})
        self.assertEqual(app.get("progress")[0].proto.value, 100)
        self.assertTrue(app.button(key="start_import").disabled)
        self.assertFalse(app.button(key="retry_import").disabled)
        app.button(key="retry_import").click().run()
        self.assertEqual([t["attempts"] for t in app.session_state["import_tasks"]], [1, 2])
        app.run()
        self.assertEqual([t["attempts"] for t in app.session_state["import_tasks"]], [1, 2])


    def test_new_upload_is_incremental_and_repeat_upload_skips_encoding(self):
        """改变文件选择新增文献，重新选原文也不重算已有向量。"""
        app = self.app
        for filename, content in (("first.txt", b"First"), ("second.txt", b"Second"), ("first.txt", b"First")):
            app.file_uploader[0].set_value([(filename, content, "text/plain")]).run()
            app.button(key="start_import").click().run()
            self.assertFalse(app.exception)
        task = app.session_state["import_tasks"][0]
        self.assertEqual(task["added_chunks"], 0)
        self.assertEqual(task["index_total"], 2)
        self.assertEqual(self.embeddings.document_calls, [["First"], ["Second"]])


    def test_model_error_keeps_file_and_retry_completes_index(self):
        """缺失本地模型时不误报成功，恢复模型后仅重试索引。"""
        app = self.app
        app.file_uploader[0].set_value([("retry.txt", b"Retry", "text/plain")]).run()
        with patch("src.retrieval.vector_store.get_embeddings", side_effect=FileNotFoundError("本地模型不存在")):
            app.button(key="start_import").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["import_tasks"][0]["status"], "failed")
        self.assertTrue(Path(app.session_state["import_tasks"][0]["path"]).is_file())
        with patch("src.data_loader.load_document", side_effect=AssertionError("不应重新加载")):
            app.button(key="retry_import").click().run()
        task = app.session_state["import_tasks"][0]
        self.assertEqual(task["status"], "success")
        self.assertTrue(task["indexed"])
        self.assertEqual(task["added_chunks"], 1)
        self.assertEqual(task["attempts"], 2)

    def test_vector_search_uses_persisted_index_and_top_k(self):
        """页面没有上传任务也能查旧库，展示排序、跨页来源与负相似度。"""
        chunks = [
            Document(page_content="神经网络实验", metadata={"chunk_id": "a1", "doc_id": "a",
                     "source_file": "论文A.pdf", "page_number": 2, "page_end": 3}),
            Document(page_content="农业实验", metadata={"chunk_id": "b1", "doc_id": "b",
                     "source_file": "论文B.pdf", "page_number": 1}),
            Document(page_content="反向向量实验", metadata={"chunk_id": "c1", "doc_id": "c",
                     "source_file": "论文C.pdf", "page_number": 4}),
        ]
        VectorStore().add_chunks(chunks)
        app = self.app
        self.assertEqual(app.number_input(key="vector_top_k").value, 5)
        self.assertEqual(app.session_state["import_tasks"], [])
        app.text_input(key="vector_query").set_value("神经网络")
        app.number_input(key="vector_top_k").set_value(2)
        app.button(key="vector_search").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(c.value.startswith("最近实际向量检索成功：") for c in app.sidebar.caption))
        self.assertEqual([element.value for element in app.text], ["神经网络实验", "农业实验"])
        self.assertTrue(any("1. 论文A.pdf · 余弦相似度 1.0000" in panel.label for panel in app.expander))
        self.assertIn("来源：论文A.pdf；物理页码：2–3", [element.value for element in app.caption])
        app.number_input(key="vector_top_k").set_value(10)
        app.button(key="vector_search").click().run()
        self.assertEqual(len(app.text), 3)
        self.assertTrue(any(panel.label.startswith("3. ") and "余弦相似度 -1.0000" in panel.label
                            for panel in app.expander))
        self.assertEqual(self.embeddings.query_calls, ["神经网络", "神经网络"])
        self.assertEqual(self.embeddings.document_calls, [[chunk.page_content for chunk in chunks]])


    def knowledge_rows(self, app=None):
        """按字段找到只读表格，避免依赖新增面板后的全局元素顺序。"""
        return next(table.value for table in (app or self.app).dataframe if "向量化状态" in table.value.columns)


if __name__ == "__main__":
    unittest.main()
