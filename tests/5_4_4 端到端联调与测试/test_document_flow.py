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
from src.data_loader import batch_import, create_import_tasks, load_document
from src.chunking import split_documents
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

    def test_workbench_tabs_keep_features_in_their_sections(self):
        """三个工作区分别承载问答、检索和原文管理。"""
        app = self.app
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "智能科研助理")
        self.assertTrue(any(c.value == "基于 RAG + Agent 的论文知识库问答系统" for c in app.caption))
        self.assertEqual([tab.label for tab in app.tabs], ["科研对话", "文档检索", "知识库"])
        chat, retrieval, knowledge = app.tabs
        self.assertEqual(len(chat.chat_input), 1)
        self.assertFalse(app.text_input(key="vector_query").value)
        self.assertTrue(any(s.value == "知识库管理面板" for s in knowledge.subheader))
        self.assertFalse(any(e.label == "知识库管理面板" for e in knowledge.expander))
        self.assertEqual(app.text_input(key="vector_query").label, "查询内容")
        self.assertEqual(app.text_input(key="vector_doc_id").label, "文档ID")
        self.assertEqual(app.number_input(key="vector_top_k").label, "Top-k")
        self.assertFalse(any("此处只检索文档块" in c.value or "统计本地请求日志中的实际 RAG 检索" in c.value
                             or "共享知识库按文档内容指纹汇总" in c.value for c in app.caption))
        self.assertTrue(any(s.value == "检索分数分布" for s in retrieval.subheader))
        self.assertTrue(any(s.value == "状态统计" for s in chat.subheader))
        self.assertFalse(app.get("graphviz_chart"))
        self.assertFalse(any((b.key or "") in {"run_agent", "clear_rag_chat"} for b in app.button))
        self.assertFalse(any(s.value in {"科研对话", "模块开发状态", "评阅说明"} for s in app.subheader))
        self.assertFalse(any("上传文档后增量写入本地知识库；下方 RAG 问答" in i.value for i in app.info))

    def test_knowledge_selection_switches_and_keeps_short_id(self):
        """两份原文可切换选中，删除一份后自动回落到另一份。"""
        from hashlib import sha256
        app = self.app
        first, second = b"# Paper A\n\nFirst full paragraph.", b"Paper B has a different full paragraph."
        a, b = sha256(first).hexdigest(), sha256(second).hexdigest()
        app.file_uploader[0].set_value([
            ("a.md", first, "text/markdown"), ("b.txt", second, "text/plain")]).run()
        app.button(key="start_import").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(any(s.value == "知识库文档" for s in app.sidebar.subheader))
        self.assertFalse(any((button.key or "").startswith("delete_document:") for button in app.sidebar.button))
        self.assertEqual(app.button(key=f"knowledge_document:{a}").label, "a.md")
        self.assertTrue(any(c.value == f"ID：{a[:8]}" for c in app.tabs[2].caption))
        self.assertEqual(app.session_state["knowledge_document_id"], a)
        app.button(key=f"knowledge_document:{b}").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["knowledge_document_id"], b)
        app.button(key=f"delete_document:{b}").click().run()
        app.button(key="confirm_delete_document").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["knowledge_document_id"], a)

    def test_sidebar_titles_and_upload_label_are_consistent(self):
        """三个侧栏标题同用header，上传组件更名但保持批量能力。"""
        app = self.app
        self.assertEqual([title.value for title in app.sidebar.header],
                         ["系统状态", "文档上传与管理", "对话历史管理"])
        self.assertEqual(app.file_uploader[0].label, "上传文档")
        self.assertTrue(app.file_uploader[0].proto.multiple_files)
        self.assertTrue(app.button(key="start_import").disabled)
        self.assertTrue(app.button(key="retry_import").disabled)

    def test_eight_character_document_filter_uses_full_identity(self):
        """真实上传文献的八位ID可检索，重名简写拒绝执行，完整ID仍有效。"""
        from hashlib import sha256
        app = self.app
        body = b"Transformer encoder layers"
        identifier = sha256(body).hexdigest()
        app.file_uploader[0].set_value([("paper.txt", body, "text/plain")]).run()
        app.button(key="start_import").click().run()
        app.selectbox(key="retrieval_method").set_value("BM25 关键词")
        app.text_input(key="vector_query").set_value("Transformer")
        app.text_input(key="vector_doc_id").set_value(identifier[:8])
        app.button(key="vector_search").click().run()
        self.assertFalse(app.exception)
        self.assertEqual([t.value for t in app.tabs[1].text], [body.decode()])
        self.assertTrue(any(c.value.startswith(f"文档 ID：{identifier[:8]}；块 ID：") for c in app.tabs[1].caption))
        self.assertEqual(self.knowledge_rows(app).iloc[0]["文档 ID"], identifier[:8])
        other = identifier[:8] + ("a" if identifier[8:] != "a" * 56 else "b") * 56
        VectorStore().add_chunks([Document(page_content="Transformer other paper", metadata={
            "doc_id": other, "chunk_id": "collision", "source_file": "other.txt"})])
        with patch("src.retrieval.bm25_retriever.BM25Retriever.search") as search:
            app.button(key="vector_search").click().run()
            search.assert_not_called()
        self.assertTrue(any("前8位ID对应多份文档" in e.value for e in app.tabs[1].error))
        app.text_input(key="vector_doc_id").set_value(identifier)
        app.button(key="vector_search").click().run()
        self.assertFalse(app.tabs[1].error)
        self.assertEqual([t.value for t in app.tabs[1].text], [body.decode()])

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

    def test_retry_separates_current_progress_from_cumulative_results(self):
        """六份混合导入后只重试两份，当前2/2与累计4成功2失败分别标明。"""
        app = self.app
        app.file_uploader[0].set_value([
            *[(f"good-{i}.txt", f"正文{i}".encode(), "text/plain") for i in range(4)],
            ("bad-a.txt", b"\xff", "text/plain"), ("bad-b.txt", b"\xfe", "text/plain"),
        ]).run()
        app.button(key="start_import").click().run()
        app.button(key="retry_import").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["import_progress"], {"completed": 2, "total": 2})
        self.assertEqual(app.get("progress")[0].proto.text, "本次操作已处理 2/2 份")
        captions = [row.value for row in app.sidebar.caption]
        self.assertIn("累计导入结果", captions)
        self.assertIn("累计成功 4 份，失败 2 份。", captions)
        self.assertEqual(len(app.sidebar.dataframe[0].value), 6)

    def test_oversized_upload_shows_failure_and_valid_file_still_indexes(self):
        """AppTest绕过浏览器大小限制，验证20MiB后端保护与页面失败/重试状态。"""
        app = self.app
        app.file_uploader[0].set_value([("oversized.pdf", b"x" * (20 * 1024 * 1024 + 1), "application/pdf"),
                                       ("valid.txt", b"Neural network", "text/plain")]).run()
        with patch("src.data_loader.load_document", wraps=load_document) as loader:
            app.button(key="start_import").click().run()
        self.assertFalse(app.exception)
        self.assertEqual([call.args[0].name for call in loader.call_args_list], ["valid.txt"])
        tasks = app.session_state["import_tasks"]
        self.assertEqual([task["status"] for task in tasks], ["failed", "success"])
        self.assertIn("文件过大", tasks[0]["error"])
        self.assertEqual(tasks[0]["path"], "")
        self.assertFalse(tasks[0]["indexed"])
        self.assertTrue(tasks[1]["indexed"])
        self.assertEqual(list(app.sidebar.dataframe[0].value["状态"]), ["失败", "成功"])
        self.assertFalse(app.button(key="retry_import").disabled)
        app.button(key="retry_import").click().run()
        self.assertFalse(app.exception)
        self.assertEqual([task["attempts"] for task in app.session_state["import_tasks"]], [2, 1])
        self.assertEqual(self.embeddings.document_calls, [["Neural network"]])

    def test_failed_upload_recovers_and_disables_retry(self):
        """页面通过失败按钮恢复成功，之后重试按钮禁用且文献存在。"""
        app = self.app
        app.file_uploader[0].set_value([("retry.txt", b"Retry", "text/plain")]).run()
        with patch("src.data_loader.load_document", side_effect=OSError("暂时失败")):
            app.button(key="start_import").click().run()
        self.assertEqual(app.session_state["import_tasks"][0]["status"], "failed")
        app.button(key="retry_import").click().run()
        self.assertFalse(app.exception)
        task = app.session_state["import_tasks"][0]
        self.assertEqual(task["status"], "success")
        self.assertEqual(task["attempts"], 2)
        self.assertEqual(task["error"], "")
        self.assertTrue(Path(task["path"]).is_file())
        self.assertTrue(app.button(key="retry_import").disabled)

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

    def test_retry_keeps_new_unimported_selection(self):
        """重试旧失败文件时，不清空用户新选但尚未导入的其他文件。"""
        app = self.app
        app.file_uploader[0].set_value([("old.txt", b"First", "text/plain")]).run()
        with patch("src.data_loader.load_document", side_effect=OSError("暂时失败")):
            app.button(key="start_import").click().run()
        app.file_uploader[0].set_value([("new.txt", b"Second", "text/plain")]).run()
        app.button(key="retry_import").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.file_uploader[0].value[0].name, "new.txt")
        self.assertFalse(app.button(key="start_import").disabled)
        app.button(key="start_import").click().run()
        self.assertFalse(app.file_uploader[0].value)
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
        self.assertIn("来源：论文A.pdf；第2–3页（物理页码）", [element.value for element in app.caption])
        app.number_input(key="vector_top_k").set_value(10)
        app.button(key="vector_search").click().run()
        self.assertEqual(len(app.text), 3)
        self.assertTrue(any(panel.label.startswith("3. ") and "余弦相似度 -1.0000" in panel.label
                            for panel in app.expander))
        self.assertEqual(self.embeddings.query_calls, ["神经网络", "神经网络"])
        self.assertEqual(self.embeddings.document_calls, [[chunk.page_content for chunk in chunks]])

    def test_vector_search_document_filter_and_non_pdf_locations(self):
        """文档过滤生效，Word 显示段落/表格、TXT 显示行号，不伪造页码。"""
        VectorStore().add_chunks([
            Document(page_content="Word 神经网络正文", metadata={"chunk_id": "w1", "doc_id": "word",
                     "source_file": "论文.docx", "paragraph_index": 3}),
            Document(page_content="反向表格", metadata={"chunk_id": "w2", "doc_id": "word",
                     "source_file": "论文.docx", "table_index": 2}),
            Document(page_content="农业文本", metadata={"chunk_id": "t1", "doc_id": "text",
                     "source_file": "论文.txt", "line_start": 4, "line_end": 8}),
        ])
        app = self.app
        app.text_input(key="vector_query").set_value("神经网络")
        app.text_input(key="vector_doc_id").set_value(" word ")
        app.button(key="vector_search").click().run()
        self.assertFalse(app.exception)
        self.assertEqual([element.value for element in app.text], ["Word 神经网络正文", "反向表格"])
        captions = [element.value for element in app.caption]
        self.assertIn("来源：论文.docx；段落3", captions)
        self.assertIn("来源：论文.docx；表格2", captions)
        self.assertFalse(any("物理页码" in value for value in captions))
        app.text_input(key="vector_query").set_value("农业")
        app.text_input(key="vector_doc_id").set_value("text")
        app.button(key="vector_search").click().run()
        self.assertEqual([element.value for element in app.text], ["农业文本"])
        self.assertIn("来源：论文.txt；行4–8", [element.value for element in app.caption])
        app.text_input(key="vector_doc_id").set_value("missing")
        app.button(key="vector_search").click().run()
        self.assertFalse(app.text)
        self.assertTrue(any("没有可检索的文档块" in element.value for element in app.info))
        self.assertEqual(self.embeddings.query_calls, ["神经网络", "农业"])

    def test_vector_search_blank_input_and_empty_index(self):
        """启动/空问题不初始化模型，空库明确提示且不计算查询向量。"""
        app = self.app
        with patch("src.retrieval.vector_store.get_embeddings") as model:
            app.run()
            app.text_input(key="vector_query").set_value("  ")
            app.button(key="vector_search").click().run()
            model.assert_not_called()
        self.assertFalse(app.exception)
        self.assertTrue(any("请输入查询内容" in element.value for element in app.warning))
        app.text_input(key="vector_query").set_value("神经网络")
        app.button(key="vector_search").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("没有可检索的文档块" in element.value for element in app.info))
        self.assertEqual(self.embeddings.query_calls, [])

    def test_vector_search_errors_are_visible_and_retryable(self):
        """模型/数据库失败明确报错，修复后重新提交可检索，不当作空库。"""
        VectorStore().add_chunks([Document(page_content="神经网络", metadata={
            "chunk_id": "a1", "doc_id": "a", "source_file": "论文.txt", "line_start": 1, "line_end": 1})])
        app = self.app
        app.text_input(key="vector_query").set_value("神经网络")
        for target, error in (("src.retrieval.vector_store.get_embeddings", FileNotFoundError("本地模型不存在")),
                              ("src.retrieval.vector_store.VectorStore.search", RuntimeError("数据库不可用"))):
            with self.subTest(target=target), patch(target, side_effect=error):
                app.button(key="vector_search").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any(str(error) in element.value for element in app.tabs[1].error))
            self.assertFalse(any("没有可检索的文档块" in element.value for element in app.info))
        app.button(key="vector_search").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.tabs[1].error)
        self.assertEqual([element.value for element in app.text], ["神经网络"])

    def test_bm25_search_without_model_and_with_document_filter(self):
        """页面 BM25 不加载权重，单文献负分仍展示正文、位置与 ID。"""
        VectorStore().add_chunks([Document(page_content="BatchNormalization", metadata={
            "chunk_id": "bn1", "doc_id": "bn", "source_file": "论文.pdf", "page_number": 2})])
        app = self.app
        app.selectbox(key="retrieval_method").set_value("BM25 关键词")
        app.text_input(key="vector_query").set_value("BATCHNORMALIZATION")
        app.text_input(key="vector_doc_id").set_value(" bn ")
        with patch("src.retrieval.vector_store.get_embeddings", side_effect=AssertionError("不能加载模型")) as model:
            app.button(key="vector_search").click().run()
            model.assert_not_called()
        self.assertFalse(app.exception)
        self.assertEqual([element.value for element in app.text], ["BatchNormalization"])
        self.assertTrue(any("BM25 分数 -" in panel.label for panel in app.expander))
        self.assertIn("来源：论文.pdf；第2页（物理页码）", [element.value for element in app.caption])
        self.assertEqual(self.embeddings.query_calls, [])
        app.text_input(key="vector_query").set_value("Normalization")
        app.button(key="vector_search").click().run()
        self.assertFalse(app.text)
        self.assertTrue(any("关键词无匹配" in element.value for element in app.info))

    def test_bm25_search_reflects_upload_and_delete(self):
        """上传后每次查询读取当前语料，新文档立即可查，已删除文档不再出现。"""
        app = self.app
        app.file_uploader[0].set_value([("first.txt", b"BatchNormalization", "text/plain")]).run()
        app.button(key="start_import").click().run()
        first_doc_id = app.session_state["import_tasks"][0]["documents"][0].metadata["doc_id"]
        app.selectbox(key="retrieval_method").set_value("BM25 关键词")
        app.text_input(key="vector_query").set_value("BatchNormalization")
        app.button(key="vector_search").click().run()
        self.assertEqual([element.value for element in app.text], ["BatchNormalization"])
        app.file_uploader[0].set_value([("second.txt", b"Adam", "text/plain")]).run()
        app.button(key="start_import").click().run()
        app.text_input(key="vector_query").set_value("Adam")
        app.button(key="vector_search").click().run()
        self.assertFalse(app.exception)
        self.assertEqual([element.value for element in app.text], ["Adam"])
        VectorStore().delete_document(first_doc_id)
        app.text_input(key="vector_query").set_value("BatchNormalization")
        app.button(key="vector_search").click().run()
        self.assertFalse(app.text)
        self.assertEqual(self.embeddings.document_calls, [["BatchNormalization"], ["Adam"]])
        self.assertEqual(self.embeddings.query_calls, [])

    def test_bm25_database_error_is_visible_and_retryable(self):
        """语料读取异常明确提示，恢复后重新提交可查，不静默切换检索方式。"""
        VectorStore().add_chunks([Document(page_content="Adam", metadata={
            "chunk_id": "a1", "doc_id": "a", "source_file": "论文.txt", "line_start": 1, "line_end": 1})])
        app = self.app
        app.selectbox(key="retrieval_method").set_value("BM25 关键词")
        app.text_input(key="vector_query").set_value("Adam")
        with patch("src.retrieval.vector_store.VectorStore.list_chunks", side_effect=RuntimeError("语料读取失败")):
            app.button(key="vector_search").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("语料读取失败" in element.value for element in app.tabs[1].error))
        self.assertFalse(app.text)
        app.button(key="vector_search").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.tabs[1].error)
        self.assertEqual([element.value for element in app.text], ["Adam"])
        self.assertEqual(self.embeddings.query_calls, [])

    def test_rrf_search_displays_fused_score_and_original_source(self):
        """页面切换混合检索，共同命中的块排序提高，并保留 Word 位置。"""
        VectorStore().add_chunks([
            Document(page_content="神经网络", metadata={"chunk_id": "a", "doc_id": "a",
                     "source_file": "论文A.pdf", "page_number": 1}),
            Document(page_content="BatchNormalization", metadata={"chunk_id": "b", "doc_id": "b",
                     "source_file": "论文B.docx", "paragraph_index": 3}),
            Document(page_content="农业 BatchNormalization", metadata={"chunk_id": "c", "doc_id": "c",
                     "source_file": "论文C.txt", "line_start": 1, "line_end": 1}),
        ])
        app = self.app
        app.selectbox(key="retrieval_method").set_value("RRF 混合检索")
        app.text_input(key="vector_query").set_value("BatchNormalization")
        app.number_input(key="vector_top_k").set_value(1)
        app.button(key="vector_search").click().run()
        self.assertFalse(app.exception)
        self.assertEqual([element.value for element in app.text], ["BatchNormalization"])
        self.assertTrue(any("1. 论文B.docx · RRF 分数" in panel.label for panel in app.expander))
        self.assertIn("来源：论文B.docx；段落3", [element.value for element in app.caption])
        self.assertEqual(self.embeddings.query_calls, ["BatchNormalization"])

    def test_rrf_empty_input_empty_store_and_model_error(self):
        """空问题/空库无需权重；有数据但模型失败不静默改为 BM25。"""
        app = self.app
        app.selectbox(key="retrieval_method").set_value("RRF 混合检索")
        with patch("src.retrieval.vector_store.get_embeddings", side_effect=AssertionError("不应加载模型")) as model:
            app.button(key="vector_search").click().run()
            app.text_input(key="vector_query").set_value("BERT")
            app.button(key="vector_search").click().run()
            model.assert_not_called()
        self.assertFalse(app.exception)
        self.assertFalse(app.tabs[1].error)
        self.assertFalse(app.text)
        VectorStore().add_chunks([Document(page_content="BERT", metadata={
            "chunk_id": "b", "doc_id": "b", "source_file": "论文.txt", "line_start": 1, "line_end": 1})])
        with patch("src.retrieval.vector_store.get_embeddings", side_effect=FileNotFoundError("本地模型不存在")):
            app.button(key="vector_search").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("本地模型不存在" in element.value for element in app.tabs[1].error))
        self.assertFalse(app.text)
        app.button(key="vector_search").click().run()
        self.assertFalse(app.tabs[1].error)
        self.assertEqual([element.value for element in app.text], ["BERT"])

    def test_rrf_search_after_upload_and_database_recovery(self):
        """新上传文档参加两路召回，数据库异常恢复后可查，删除过滤无陈旧块。"""
        app = self.app
        for filename, text in (("first.txt", b"BERT"), ("second.txt", b"Reranker")):
            app.file_uploader[0].set_value([(filename, text, "text/plain")]).run()
            app.button(key="start_import").click().run()
        doc_id = app.session_state["import_tasks"][0]["documents"][0].metadata["doc_id"]
        app.selectbox(key="retrieval_method").set_value("RRF 混合检索")
        app.text_input(key="vector_query").set_value("Reranker")
        app.number_input(key="vector_top_k").set_value(1)
        with patch("src.retrieval.vector_store.VectorStore.list_chunks", side_effect=RuntimeError("语料读取失败")):
            app.button(key="vector_search").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("语料读取失败" in element.value for element in app.tabs[1].error))
        self.assertFalse(app.text)
        app.button(key="vector_search").click().run()
        self.assertFalse(app.tabs[1].error)
        self.assertEqual([element.value for element in app.text], ["Reranker"])
        VectorStore().delete_document(doc_id)
        app.text_input(key="vector_doc_id").set_value(doc_id)
        app.button(key="vector_search").click().run()
        self.assertFalse(app.text)
        self.assertEqual(self.embeddings.document_calls, [["BERT"], ["Reranker"]])

    def test_model_reranking_displays_new_order_score_and_sources(self):
        """页面使用模型分数重新排列，保留 Word 段落与 TXT 行范围。"""
        VectorStore().add_chunks([
            Document(page_content="神经网络", metadata={"chunk_id": "a", "doc_id": "a",
                     "source_file": "论文A.pdf", "page_number": 1}),
            Document(page_content="BatchNormalization", metadata={"chunk_id": "b", "doc_id": "b",
                     "source_file": "论文B.docx", "paragraph_index": 3}),
            Document(page_content="农业 BatchNormalization", metadata={"chunk_id": "c", "doc_id": "c",
                     "source_file": "论文C.txt", "line_start": 2, "line_end": 3}),
        ])
        app = self.app
        app.selectbox(key="retrieval_method").set_value("RRF + 模型重排")
        app.text_input(key="vector_query").set_value("BatchNormalization")
        app.number_input(key="vector_top_k").set_value(2)
        with patch("src.retrieval.reranker.get_reranker") as model:
            # 按正文给出明确测试分数，模型无需与实际论文质量等同。
            values = {"神经网络": 0.1, "BatchNormalization": 0.8, "农业 BatchNormalization": 0.9}
            model.return_value.predict.side_effect = lambda pairs, **kwargs: [values[text] for _, text in pairs]
            app.button(key="vector_search").click().run()
            self.assertEqual(len(model.return_value.predict.call_args.args[0]), 3)
        self.assertFalse(app.exception)
        self.assertEqual([element.value for element in app.text], ["农业 BatchNormalization", "BatchNormalization"])
        self.assertTrue(any("论文C.txt · 模型相关性分数 0.9000" in panel.label for panel in app.expander))
        self.assertIn("来源：论文C.txt；行2–3", [element.value for element in app.caption])
        self.assertIn("来源：论文B.docx；段落3", [element.value for element in app.caption])

    def test_model_reranking_empty_cases_and_error_recovery(self):
        """空问题/空库无需模型，模型缺失或推理失败有错误提示，修复后可重新提交。"""
        app = self.app
        app.selectbox(key="retrieval_method").set_value("RRF + 模型重排")
        with patch("src.retrieval.reranker.get_reranker", side_effect=AssertionError("不应加载模型")) as model:
            app.button(key="vector_search").click().run()
            app.text_input(key="vector_query").set_value("BERT")
            app.button(key="vector_search").click().run()
            model.assert_not_called()
        self.assertFalse(app.tabs[1].error)
        VectorStore().add_chunks([Document(page_content="BERT", metadata={
            "chunk_id": "b", "doc_id": "b", "source_file": "论文.txt", "line_start": 1, "line_end": 1})])
        for message in ("重排模型不存在", "重排推理失败"):
            with patch("src.retrieval.reranker.get_reranker", side_effect=RuntimeError(message)):
                app.button(key="vector_search").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any(message in element.value for element in app.tabs[1].error))
            self.assertFalse(app.text)
        with patch("src.retrieval.reranker.get_reranker") as model:
            model.return_value.predict.return_value = [0.8]
            app.button(key="vector_search").click().run()
        self.assertFalse(app.tabs[1].error)
        self.assertEqual([element.value for element in app.text], ["BERT"])

    def test_model_reranking_uploaded_document_filter_and_delete(self):
        """新文献可精排，文档 ID 限定在模型评分前生效，删除后不加载重排模型。"""
        app = self.app
        for filename, text in (("first.txt", b"BERT"), ("second.txt", b"Reranker")):
            app.file_uploader[0].set_value([(filename, text, "text/plain")]).run()
            app.button(key="start_import").click().run()
        doc_id = app.session_state["import_tasks"][0]["documents"][0].metadata["doc_id"]
        app.selectbox(key="retrieval_method").set_value("RRF + 模型重排")
        app.text_input(key="vector_query").set_value("Reranker")
        app.text_input(key="vector_doc_id").set_value(doc_id)
        with patch("src.retrieval.reranker.get_reranker") as model:
            model.return_value.predict.return_value = [0.7]
            app.button(key="vector_search").click().run()
            self.assertEqual([element.value for element in app.text], ["Reranker"])
            model.return_value.predict.assert_called_once_with(
                [["Reranker", "Reranker"]], batch_size=8, show_progress_bar=False)
            VectorStore().delete_document(doc_id)
            model.reset_mock()
            app.button(key="vector_search").click().run()
            self.assertFalse(app.text)
            model.assert_not_called()
        self.assertFalse(app.exception)
        self.assertEqual(self.embeddings.document_calls, [["BERT"], ["Reranker"]])


    def test_layout_library_delete_cancel_restore(self):
        """真实页面上传、取消删除、删除和恢复均核验原文及Chroma。"""
        from src.frontend.components.documents import list_documents
        app = self.app
        raw, index = Path(self.directory.name) / "raw", Path(self.directory.name) / "index"
        app.file_uploader[0].set_value([("paper.md", b"Transformer uses six layers.", "text/markdown")]).run()
        app.button(key="start_import").click().run()
        doc_id = list_documents(raw, index)[0]["doc_id"]
        self.assertEqual([tab.label for tab in app.tabs], ["科研对话", "文档检索", "知识库"])
        self.assertEqual(app.sidebar.get("progress")[0].proto.value, 100)
        self.assertEqual(self.delete_button(app, doc_id).label, "删除")
        self.delete_button(app).click().run()
        self.assertFalse(any(b.key == f"delete_document:{doc_id}" for b in app.tabs[2].button))
        self.assertEqual(app.button(key="confirm_delete_document").label, "确认删除")
        self.assertEqual(app.button(key="cancel_delete_document").label, "取消")
        self.assertFalse(any("待删除" in row.value for row in app.tabs[2].warning))
        self.assertEqual(VectorStore().count(), 1)
        app.button(key="cancel_delete_document").click().run()
        self.assertTrue((raw / doc_id / "paper.md").is_file())
        self.delete_button(app).click().run()
        app.button(key="confirm_delete_document").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(VectorStore().count(), 0)
        self.assertFalse((raw / doc_id).exists())
        self.assertEqual(app.session_state["import_tasks"], [])
        self.assertTrue(app.button(key="start_import").disabled)
        app.run()
        self.assertEqual(VectorStore().count(), 0)

    def test_delete_only_selected_document_keeps_other_index_and_progress(self):
        """删除一份文献不会清空全库，保留任务的进度与真实列表一致。"""
        app = self.app
        app.file_uploader[0].set_value([("a.txt", b"Adam", "text/plain"),
                                       ("b.txt", b"Transformer", "text/plain")]).run()
        app.button(key="start_import").click().run()
        self.delete_button(app).click().run()
        app.button(key="confirm_delete_document").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(VectorStore().count(), 1)
        self.assertEqual(VectorStore().list_chunks()[0].page_content, "Transformer")
        self.assertEqual(app.session_state["import_progress"], {"completed": 1, "total": 1})
        self.assertEqual(sum((b.key or "").startswith("delete_document:") for b in app.tabs[2].button), 1)
        app.selectbox(key="retrieval_method").set_value("BM25 关键词")
        app.text_input(key="vector_query").set_value("Adam")
        app.button(key="vector_search").click().run()
        self.assertFalse(app.text)

    def test_library_persists_without_upload_and_failure_keeps_original(self):
        """页面重载读全库；索引删除失败会回滚原文且允许重试。"""
        app = self.app
        app.file_uploader[0].set_value([("keep.txt", b"Adam", "text/plain")]).run()
        app.button(key="start_import").click().run()
        app.session_state["import_tasks"] = []
        app.file_uploader[0].set_value([]).run()
        from src.frontend.components.documents import list_documents
        doc_id = list_documents(Path(self.directory.name) / "raw", Path(self.directory.name) / "index")[0]["doc_id"]
        self.delete_button(app).click().run()
        with patch("src.retrieval.vector_store.VectorStore.delete_document", side_effect=RuntimeError("索引删除失败")):
            app.button(key="confirm_delete_document").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("索引删除失败" in e.value for e in app.error))
        self.assertEqual(VectorStore().count(), 1)
        self.assertTrue((Path(self.directory.name) / "raw" / doc_id / "keep.txt").exists())
        app.button(key="confirm_delete_document").click().run()
        self.assertEqual(VectorStore().count(), 0)

    def knowledge_rows(self, app=None):
        """按字段找到只读表格，避免依赖新增面板后的全局元素顺序。"""
        return next(table.value for table in (app or self.app).dataframe if "向量化状态" in table.value.columns)

    def test_knowledge_panel_empty_is_read_only(self):
        """空库显示真实零值，刷新不会建向量库、编码或改变当前会话。"""
        app = self.app
        self.assertEqual({m.label: m.value for m in app.metric if m.label.startswith("知识库")
                          or m.label == "已向量化文档数"},
                         {"知识库文档数": "0", "已向量化文档数": "0", "知识库索引块数": "0"})
        session = app.session_state["agent_session_id"]
        app.button(key="refresh_knowledge").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["agent_session_id"], session)
        self.assertFalse((Path(self.directory.name) / "index").exists())
        self.assertEqual(self.embeddings.document_calls, [])

    def test_knowledge_panel_tracks_failed_unindexed_and_refresh(self):
        """成功与写入失败并存，刷新后磁盘状态仍准确但不补造本页批次结果。"""
        from streamlit.testing.v1 import AppTest
        app = self.app
        app.file_uploader[0].set_value([("ok.txt", b"Adam", "text/plain"),
                                       ("failed.txt", b"Failure", "text/plain")]).run()
        original = VectorStore.add_chunks
        def write_or_fail(store, chunks):
            if chunks[0].metadata["source_file"] == "failed.txt":
                raise OSError("测试索引写入失败")
            return original(store, chunks)
        with patch.object(VectorStore, "add_chunks", autospec=True, side_effect=write_or_fail):
            app.button(key="start_import").click().run()
        rows = self.knowledge_rows(app).set_index("文件名")
        self.assertEqual(rows.loc["ok.txt", "向量化状态"], "已向量化")
        self.assertEqual(rows.loc["failed.txt", "向量化状态"], "未向量化")
        self.assertEqual(set(rows.columns), {"文档 ID", "原文状态", "向量化状态", "索引块数"})
        self.assertTrue(any("索引写入失败" in c.value for c in app.sidebar.caption))
        self.assertEqual({m.label: m.value for m in app.metric if m.label.startswith("知识库")
                          or m.label == "已向量化文档数"},
                         {"知识库文档数": "2", "已向量化文档数": "1", "知识库索引块数": "1"})
        calls = len(self.embeddings.document_calls)
        restored = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "src/frontend/app.py"),
                                     default_timeout=10).run()
        rows = self.knowledge_rows(restored)
        self.assertEqual(list(rows.columns), ["文件名", "文档 ID", "原文状态", "向量化状态", "索引块数"])
        self.assertEqual(set(rows["向量化状态"]), {"已向量化", "未向量化"})
        self.assertEqual(len(self.embeddings.document_calls), calls)
        app.button(key="retry_import").click().run()
        rows = self.knowledge_rows(app)
        self.assertEqual(set(rows["向量化状态"]), {"已向量化"})
        self.assertTrue(all(task["indexed"] for task in app.session_state["import_tasks"]))

    def test_knowledge_panel_partial_index_does_not_claim_import_success(self):
        """第二批失败时面板显示实际500块，左侧保留失败详情，重试补全。"""
        import hashlib
        from streamlit.testing.v1 import AppTest
        app = self.app
        data = b"Paper blocks"
        doc_id = hashlib.sha256(data).hexdigest()
        chunks = [Document(page_content=f"block {i}", metadata={"doc_id": doc_id,
                  "chunk_id": f"block-{i}", "source_file": "partial.txt"}) for i in range(501)]
        app.file_uploader[0].set_value([("partial.txt", data, "text/plain")]).run()
        original = VectorStore.add_chunks
        def fail_last_batch(store, batch):
            if len(batch) == 1:
                raise TimeoutError("第二批失败")
            return original(store, batch)
        with patch("src.chunking.split_documents", return_value=chunks), \
                patch.object(VectorStore, "add_chunks", autospec=True, side_effect=fail_last_batch):
            app.button(key="start_import").click().run()
        row = self.knowledge_rows(app).iloc[0]
        self.assertEqual(row["索引块数"], 500)
        self.assertEqual(row["向量化状态"], "部分入库")
        restored = AppTest.from_file(str(ROOT / "src/frontend/app.py"), default_timeout=10).run()
        self.assertEqual(self.knowledge_rows(restored).iloc[0]["向量化状态"], "部分入库")
        task = app.session_state["import_tasks"][0]
        self.assertEqual((task["status"], task["chunk_count"]), ("failed", 501))
        self.assertTrue(any("第二批失败" in c.value for c in app.sidebar.caption))
        with patch("src.chunking.split_documents", return_value=chunks):
            app.button(key="retry_import").click().run()
        self.assertEqual(self.knowledge_rows(app).iloc[0]["索引块数"], 501)
        self.assertEqual(self.knowledge_rows(app).iloc[0]["向量化状态"], "已向量化")
        self.assertEqual(len(self.embeddings.document_calls[-1]), 1)

    def test_knowledge_panel_missing_source_and_external_index_change(self):
        """只剩索引时明确原文缺失；外部移除块后刷新显示未向量化，不沿用旧批次成功。"""
        app = self.app
        app.file_uploader[0].set_value([("paper.txt", b"BERT", "text/plain")]).run()
        app.button(key="start_import").click().run()
        task = app.session_state["import_tasks"][0]
        source = Path(task["path"])
        source.unlink()
        app.button(key="refresh_knowledge").click().run()
        row = self.knowledge_rows(app).iloc[0]
        self.assertEqual(row["原文状态"], "缺失")
        self.assertEqual(row["向量化状态"], "已向量化")
        self.assertTrue(self.delete_button(app).disabled)
        source.write_bytes(b"BERT")
        VectorStore().delete_document(source.parent.name)
        app.button(key="refresh_knowledge").click().run()
        row = self.knowledge_rows(app).iloc[0]
        self.assertEqual(row["向量化状态"], "未向量化")
        self.assertEqual(row["索引块数"], 0)
        self.assertTrue(app.session_state["import_tasks"][0]["indexed"])  # 旧批次不能冒充当前索引状态。

    def test_knowledge_panel_content_alias_keeps_both_batch_results(self):
        """同内容异名按指纹合并，仍保留一个成功和另一个失败的实际结果。"""
        app = self.app
        app.file_uploader[0].set_value([("a.txt", b"BERT", "text/plain"),
                                       ("b.txt", b"BERT", "text/plain")]).run()
        original = VectorStore.add_chunks
        def fail_alias(store, chunks):
            if chunks[0].metadata["source_file"] == "b.txt":
                raise OSError("第二个别名失败")
            return original(store, chunks)
        with patch.object(VectorStore, "add_chunks", autospec=True, side_effect=fail_alias):
            app.button(key="start_import").click().run()
        rows = self.knowledge_rows(app)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["文件名"], "a.txt / b.txt")
        self.assertEqual(rows.iloc[0]["索引块数"], 1)
        self.assertEqual([task["status"] for task in app.session_state["import_tasks"]], ["success", "failed"])
        self.assertTrue(any("第二个别名失败" in c.value for c in app.sidebar.caption))

    def test_knowledge_panel_index_error_is_unknown_not_zero(self):
        """索引不可读不显示正常空库或捏造零统计，修复后刷新重新读取。"""
        app = self.app
        app.file_uploader[0].set_value([("paper.txt", b"BERT", "text/plain")]).run()
        app.button(key="start_import").click().run()
        with patch("src.frontend.components.documents.VectorStore.list_chunks", side_effect=RuntimeError("索引不可读")):
            app.button(key="refresh_knowledge").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("索引不可读" in e.value for e in app.error))
        self.assertFalse([m for m in app.metric if m.label.startswith("知识库")])
        self.assertFalse([d for d in app.dataframe if "向量化状态" in d.value.columns])
        app.button(key="refresh_knowledge").click().run()
        self.assertEqual(self.knowledge_rows(app).iloc[0]["索引块数"], 1)


if __name__ == "__main__":
    unittest.main()
