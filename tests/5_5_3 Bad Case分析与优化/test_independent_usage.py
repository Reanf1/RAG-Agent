"""独立审计缺陷回归：移植原预期行为断言；模型报文和微型向量为明确替身。"""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from io import BytesIO
import json
import unittest
from unittest.mock import patch
from urllib.request import Request
from langchain_core.tools import tool
from src.agent.react_loop import act
from src.agent.react_loop import run_react
from src.agent.router import execute_calls
from src.agent.tools import calculator, execute_tool, keyword_extract
from src.agent.tools import _record_tool_usage
from src.utils.config import load_config
from src.utils.logger import update_agent_metrics

class ActionFailureUsageAudit(unittest.TestCase):

    def test_completed_model_usage_survives_batch_count_validation_error(self):
        """已收到的真实用量应随错误事件保留；当前实现丢弃此字段。"""

        @tool
        def knowledge_base_search(question: str) -> dict:
            """测试工具：不完整的批次不得开始执行。"""
            self.fail('不完整调用批次不应执行工具')
        usage = {'prompt_eval_count': 1615, 'eval_count': 42}
        response = {'model': 'qwen2.5:7b', 'done': True, 'done_reason': 'stop', **usage, 'message': {'content': '', 'tool_calls': [{'function': {'name': 'knowledge_base_search', 'arguments': {'question': 'T2T-ViT与CaiT的架构变化'}}}]}}
        thought = {'next_step': 'tool', 'tool_name': 'knowledge_base_search', 'parallel_tools': ['knowledge_base_search', 'knowledge_base_search']}
        with patch('src.agent.react_loop._model_request', return_value=Request('http://localhost/api/chat')), patch('src.agent.react_loop.urlopen', return_value=BytesIO(json.dumps(response).encode('utf-8'))):
            events = list(act('比较T2T-ViT与CaiT', thought, [knowledge_base_search]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['type'], 'error')
        self.assertIn('未调用完整的独立工具批次', events[0]['message'])
        self.assertEqual(events[0].get('usage'), usage)

    def test_keyword_validation_failure_preserves_already_reported_usage(self):
        """S08同类：拒绝扩展词正确，但已结束模型的用量不能随结果丢弃。"""
        response = {'model': '明确报文替身', 'prompt_eval_count': 100, 'eval_count': 42, 'message': {'content': json.dumps({'keywords': ['image patches']})}}
        with patch('src.agent.tools._tool_model_response', return_value=response):
            event = execute_tool('keyword_extract', {'text': 'split an image into patches'}, [keyword_extract])
        self.assertEqual(event['status'], 'error')
        self.assertIn('关键词不在输入原文', event['error'])
        metrics = update_agent_metrics({}, {**event, 'iteration': 1})
        self.assertEqual(metrics['tokens']['known_total'], 142)

    def test_calculator_input_error_has_known_zero_model_usage(self):
        """I05同类：纯四则工具输入校验失败没有LLM调用，应保留确定的零。"""
        with patch('src.agent.tools._tool_model_response') as model:
            event = execute_tool('calculator', {'expression': '2**3'}, [calculator])
        model.assert_not_called()
        self.assertEqual(event['status'], 'error')
        self.assertEqual(event['error_kind'], 'input')
        metrics = update_agent_metrics({}, {**event, 'iteration': 1})
        self.assertEqual(metrics['tokens']['total'], 0)
        self.assertEqual(metrics['tokens']['unknown_calls'], 0)

    def test_invalid_thought_and_observation_keep_completed_response_usage(self):
        """参数校验失败发生在响应之后；两阶段均保留服务返回的真实账目。"""
        for question in ("帮我处理一下", "什么是深度学习？"):
            packet = {"model": "报文替身", "done": True, "done_reason": "stop", "prompt_eval_count": 90,
                      "eval_count": 10, "message": {"content": "不是合法JSON"}}
            with self.subTest(question=question), patch("src.agent.react_loop.urlopen", return_value=BytesIO(json.dumps(packet).encode())):
                events = list(run_react(question, []))
                error = next(event for event in events if event["type"] == "error")
                self.assertEqual(error["usage"], {"prompt_eval_count": 90, "eval_count": 10})
                self.assertEqual(events[-1]["metrics"]["tokens"]["total"], 100)

    def test_retry_deadline_preserves_first_completed_usage_and_marks_unknown_tail(self):
        """第二次仍运行时结束等待，保留第一次失败的用量，不接收迟到响应。"""
        from threading import Event
        from time import sleep
        calls, exited = [], Event()

        @tool
        def probe() -> dict:
            """真实线程、用量报文替身；第一次可重试超时，第二次超过等待上限。"""
            calls.append(1)
            if len(calls) == 1:
                _record_tool_usage({"prompt_eval_count": 100, "eval_count": 20})
                raise TimeoutError("第一次已返回模型响应后失败")
            try:
                sleep(.15)
                return {"usage": {"prompt_eval_count": 999, "eval_count": 999}}
            finally:
                exited.set()

        config = load_config()
        config["agent"].update(tool_timeout_seconds=.07, max_tool_retries=1)
        with patch("src.agent.router.load_config", return_value=config):
            events = list(execute_calls([{"call_id": "retry-probe", "name": "probe", "args": {}}], [probe]))
        self.assertTrue(exited.wait(1))
        event = events[0]
        self.assertEqual(event["error_kind"], "deadline")
        self.assertEqual(event["usage"], {"prompt_eval_count": 100, "eval_count": 20})
        self.assertTrue(event["usage_incomplete"])
        metrics = update_agent_metrics({}, {**event, "iteration": 1})
        self.assertEqual(metrics["tokens"]["known_total"], 120)
        self.assertIsNone(metrics["tokens"]["total"])

if __name__ == "__main__":
    unittest.main()
