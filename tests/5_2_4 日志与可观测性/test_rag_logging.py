"""5.2.4 日志与可观测性：TestRAGLogging。"""

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
from src.utils.logger import record_rag_request


class TestRAGLogging(unittest.TestCase):
    """实际临时 JSONL 文件与确定数据验证日志；不把样例用量当作模型实测。"""

    def setUp(self):
        from src.utils.config import load_config
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = load_config()
        self.config["paths"]["logs"] = self.directory.name
        patcher = patch("src.utils.logger.load_config", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.message = {"request_id": "request-1", "session_id": "session-1", "question": "论文结论是什么？",
                        "started_at": "2026-09-30T12:00:00+08:00",
                        "request_info": {"llm": self.config["llm"], "retrieval": self.config["retrieval"],
                                         "prompt_version": "rag-v3"},
                        "retrieval_status": "success", "retrieved_documents": [
                            {"rank": 1, "text": "科研原文\n公式 $x^2$。", "score": 0.8,
                             "metadata": {"source_file": "中文.pdf", "page_number": 3, "chunk_id": "块1"}}],
                        "generation_attempted": True, "answer": "结论[参考文档1]", "raw_answer": "原始回答",
                        "usage": {"prompt_eval_count": 400, "eval_count": 20},
                        "retrieval_seconds": 0.2, "generation_seconds": 2.0, "elapsed_seconds": 2.3}

    def test_full_utf8_record_preserves_original_text_metadata_and_actual_usage(self):
        self.message["retrieved_documents"][0]["text"] *= 2000
        before = deepcopy(self.message)
        record_rag_request(self.message, "completed")
        paths = list(Path(self.directory.name).glob("rag_*.jsonl"))
        self.assertEqual(len(paths), 1)
        raw = paths[0].read_text(encoding="utf-8")
        self.assertIn("科研原文", raw)
        self.assertEqual(len(raw.splitlines()), 1)  # 正文换行编码为 JSON 转义。
        record = json.loads(raw)
        self.assertEqual(record["question"], self.message["question"])
        self.assertEqual(record["retrieval"]["documents"], self.message["retrieved_documents"])
        self.assertEqual(record["tokens"], {"input": 400, "output": 20, "total": 420, "source": "ollama"})
        self.assertEqual(record["timing"], {"retrieval_seconds": 0.2, "generation_seconds": 2.0, "response_seconds": 2.3})
        self.assertEqual(record["raw_answer"], "原始回答")
        self.assertTrue(record["timestamp"].endswith("+08:00"))
        self.assertEqual(self.message, before)


if __name__ == "__main__":
    unittest.main()
