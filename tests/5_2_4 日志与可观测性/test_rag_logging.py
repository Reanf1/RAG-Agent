"""5.2.4 日志与可观测性：TestRAGLogging。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
import unittest
from unittest.mock import patch
from src.utils.logger import read_rag_requests, record_agent_request, record_rag_request, retrieval_score_distribution


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

    def test_unavailable_tokens_remain_unknown_and_not_called_is_zero(self):
        self.message.pop("usage")
        failed = record_rag_request(self.message, "error")
        self.assertEqual(failed["tokens"], {"input": None, "output": None, "total": None, "source": "unavailable"})
        pending = record_rag_request({**self.message, "generation_attempted": False}, "awaiting_confirmation")
        self.assertEqual(pending["tokens"]["total"], 0)
        self.assertEqual(pending["tokens"]["source"], "not_called")
        cached = record_rag_request({**self.message, "cache": {"hit": True},
                                     "original_usage": {"prompt_eval_count": 400, "eval_count": 20}}, "completed")
        self.assertEqual(cached["tokens"]["total"], 0)
        self.assertEqual(cached["original_usage"]["eval_count"], 20)

    def test_daily_files_and_confirmation_snapshots_count_one_request(self):
        with patch("src.utils.logger.request_time", side_effect=["2026-09-30T23:59:59+08:00", "2026-10-01T00:00:01+08:00"]):
            record_rag_request(self.message, "awaiting_confirmation")
            record_rag_request(self.message, "completed")
        self.assertEqual(len(list(Path(self.directory.name).glob("rag_*.jsonl"))), 2)
        records, invalid = read_rag_requests()
        self.assertEqual(invalid, 0)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "completed")
        self.assertEqual(retrieval_score_distribution()["distributions"][0]["count"], 1)

    def test_histogram_boundaries_and_model_revisions_are_separate(self):
        scores = [0, 0.0999, 0.1, 0.3, 0.7, 0.999, 1.0]
        for index, score in enumerate(scores):
            message = deepcopy(self.message)
            message["request_id"] = f"sample-{index}"
            message["retrieved_documents"][0]["score"] = score
            record_rag_request(message, "completed")
        other = deepcopy(self.message)
        other["request_id"] = "new-model"
        other["request_info"]["retrieval"]["reranker_revision"] = "another-revision"
        record_rag_request(other, "completed")
        summary = retrieval_score_distribution()
        group = next(item for item in summary["distributions"] if item["revision"] == self.config["retrieval"]["reranker_revision"])
        self.assertEqual([row["count"] for row in group["bins"]], [2, 1, 0, 1, 0, 0, 0, 1, 0, 2])
        self.assertAlmostEqual(group["mean"], sum(scores) / 7)
        self.assertEqual((group["min"], group["max"]), (0, 1))
        self.assertEqual(len(summary["distributions"]), 2)

    def test_empty_failed_and_cache_requests_do_not_add_zero_score_samples(self):
        for index, status in enumerate(("empty", "error", "skipped_cache")):
            message = {**self.message, "request_id": str(index), "retrieved_documents": [],
                       "retrieval_status": status, "cache": {"hit": status == "skipped_cache"}}
            record_rag_request(message, "error" if status == "error" else "completed")
        summary = retrieval_score_distribution()
        self.assertEqual(summary["requests"], 3)
        self.assertEqual(summary["distributions"], [])
        self.assertEqual((summary["empty_retrievals"], summary["failed_retrievals"], summary["cache_hits"]), (1, 1, 1))

    def test_corrupted_tail_is_preserved_and_next_record_remains_readable(self):
        record_rag_request(self.message, "started")
        path = next(Path(self.directory.name).glob("rag_*.jsonl"))
        with path.open("ab") as output:
            output.write(b'{"broken":')
        record_rag_request({**self.message, "request_id": "next"}, "completed")
        records, invalid = read_rag_requests()
        self.assertEqual(invalid, 1)
        self.assertEqual({row["request_id"] for row in records}, {"request-1", "next"})
        self.assertIn(b'{"broken":\n', path.read_bytes())
        self.assertEqual(retrieval_score_distribution()["invalid_lines"], 1)
        with path.open("ab") as output:
            output.write(b'{"schema_version":1,"request_id":"invalid","retrieval":{"status":"success","top1_score":0.5}}\n')
        self.assertEqual(retrieval_score_distribution()["invalid_lines"], 2)

    def test_threaded_appends_keep_complete_independent_lines(self):
        def write(index):
            return record_rag_request({**self.message, "request_id": f"thread-{index}"}, "completed")
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(write, range(32)))
        records, invalid = read_rag_requests()
        self.assertEqual((len(records), invalid), (32, 0))
        self.assertEqual(retrieval_score_distribution()["distributions"][0]["count"], 32)

    def test_agent_corrupted_tail_preserves_new_record_and_rejects_invalid_data(self):
        """共用写入器后，Agent也必须隔开断电半行，非法快照不能损坏文件。"""
        event = {"request_id": "agent-1", "user_id": "alice", "session_id": "session-1",
                 "type": "done", "metrics": {"tokens": {"total": 20}}}
        before = deepcopy(event)
        record_agent_request("中文问题\n第二行", event)
        path = next(Path(self.directory.name).glob("agent_*.jsonl"))
        with path.open("ab") as output:
            output.write(b'{"broken":')
        record_agent_request("新问题", {**event, "request_id": "agent-2"})
        lines = path.read_bytes().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[1], b'{"broken":')
        self.assertEqual(json.loads(lines[0])["question"], "中文问题\n第二行")
        self.assertEqual(json.loads(lines[2])["request_id"], "agent-2")
        saved = path.read_bytes()
        with self.assertRaises(ValueError):
            record_agent_request("非法指标", {**event, "metrics": {"seconds": float("nan")}})
        self.assertEqual(path.read_bytes(), saved)
        self.assertEqual(event, before)

    def test_serialization_failure_does_not_damage_previous_records(self):
        record_rag_request(self.message, "completed")
        self.message["retrieved_documents"][0]["score"] = float("nan")
        with self.assertRaises(ValueError):
            record_rag_request(self.message, "error")
        records, invalid = read_rag_requests()
        self.assertEqual((len(records), invalid), (1, 0))
        self.assertEqual(records[0]["retrieval"]["top1_score"], 0.8)


if __name__ == "__main__":
    unittest.main()
