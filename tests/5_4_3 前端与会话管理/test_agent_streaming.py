"""Agent真实分包协议测试：流式请求、校验、断流和实际用量。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from io import BytesIO
import json
import unittest
from unittest.mock import patch
from src.agent.react_loop import _observe_events, run_react
from tests.helpers import StreamingResponse


class TestAgentStreaming(unittest.TestCase):
    def packets(self, decision=None):
        value = decision or {"observation": "可以回答。", "decision": "finish", "task_complete": True,
                             "answer": "中文回答\n含引号\"与路径\\。"}
        text = json.dumps(value, ensure_ascii=False)
        return [{"message": {"content": text[i:i + 3]}, "done": False} for i in range(0, len(text), 3)] + [
            {"message": {"content": ""}, "done": True, "done_reason": "stop", "model": "qwen2.5:7b",
             "prompt_eval_count": 123, "eval_count": 45}]

    def test_observation_emits_single_final_event(self):
        """Observation只在解析校验后产出一条结论，不再逐字暴露JSON片段。"""
        with patch("src.agent.react_loop.urlopen", return_value=StreamingResponse(self.packets())):
            events = list(_observe_events("解释概念", [], stream=True))
        self.assertEqual([event["type"] for event in events], ["observation"])
        self.assertEqual(events[0]["answer"], '中文回答\n含引号"与路径\\。')

    def test_response_closes_and_reports_real_usage(self):
        response = StreamingResponse(self.packets())
        with patch("src.agent.react_loop.urlopen", return_value=response) as http:
            events = list(_observe_events("解释概念", [], stream=True))
        self.assertTrue(response.closed)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["answer"], '中文回答\n含引号"与路径\\。')
        self.assertEqual(events[0]["usage"], {"prompt_eval_count": 123, "eval_count": 45})
        self.assertTrue(json.loads(http.call_args.args[0].data)["stream"])

    def test_disconnect_keeps_partial_but_does_not_complete(self):
        with patch("src.agent.react_loop.urlopen", return_value=StreamingResponse(self.packets()[:-1])):
            events = list(run_react("什么是深度学习？", [], stream=True))
        self.assertFalse(events[-1]["task_complete"])
        self.assertIn("中文回答", events[-1]["full_response"])
        self.assertIn("回答未完成", events[-1]["full_response"])
        self.assertIsNone(events[-1]["metrics"]["tokens"]["total"])

    def test_null_message_keeps_partial_and_closes_response(self):
        """损坏的分包不能抛出未处理异常，也不能把已输出片段当完整答案。"""
        for final in (False, True):
            packets = self.packets()[:-1] + [{"message": None, "done": final}]
            response = StreamingResponse(packets)
            with self.subTest(final=final), patch("src.agent.react_loop.urlopen", return_value=response):
                events = list(run_react("什么是深度学习？", [], stream=True))
            self.assertTrue(response.closed)
            self.assertFalse(events[-1]["task_complete"])
            self.assertIn("回答未完成", events[-1]["full_response"])

    def test_stream_tokens_do_not_duplicate_token_cost_or_trace(self):
        with patch("src.agent.react_loop.urlopen", return_value=StreamingResponse(self.packets())):
            events = list(run_react("什么是深度学习？", [], stream=True))
        done = events[-1]
        self.assertTrue(done["task_complete"])
        self.assertEqual(done["metrics"]["tokens"]["total"], 168)
        self.assertEqual(sum(c["phase"] == "Observation" for c in done["metrics"]["calls"]), 1)
        self.assertFalse(any(t["type"] == "token" for t in done["metrics"]["trace"]))

    def test_length_cutoff_and_invalid_finish_are_not_success(self):
        for mode in ("length", "invalid"):
            packets = self.packets()
            if mode == "length":
                packets[-1]["done_reason"] = "length"
            else:
                packets = self.packets({"observation": "不足", "decision": "continue", "task_complete": True, "answer": "错"})
            with self.subTest(mode=mode), patch("src.agent.react_loop.urlopen", return_value=StreamingResponse(packets)):
                events = list(run_react("什么是深度学习？", [], stream=True))
            self.assertFalse(events[-1]["task_complete"])

    def test_close_during_stream_closes_network_without_final_event(self):
        response = StreamingResponse(self.packets())
        with patch("src.agent.react_loop.urlopen", return_value=response):
            events = _observe_events("概念", [], stream=True)
            next(events)  # 触发生成器启动与网络请求。
            events.close()
        self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()
