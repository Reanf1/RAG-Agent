"""5.4.2 可观测性与健康检查：TestHealthCheck、TestHealthCheckPage。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from copy import deepcopy
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


if __name__ == "__main__":
    unittest.main()
