"""独立审计缺陷回归：移植原预期行为断言；模型报文和微型向量为明确替身。"""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from contextlib import ExitStack
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import Request
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from src.agent.react_loop import act, observe
from src.agent.router import execute_calls
from src.utils.config import load_config

class AgentAudit(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix='agent-audit-'))
        self.directory = Path(directory)
        self.config = deepcopy(load_config())
        self.config['paths'].update(logs=str(self.directory / 'logs'), raw_documents=str(self.directory / 'raw'), vector_index=str(self.directory / 'index'))
        self.config['agent'].update(max_parallel_calls=2, tool_timeout_seconds=1, max_tool_retries=1, max_repeated_calls=2)
        for module in ('src.agent.memory', 'src.agent.react_loop', 'src.agent.router', 'src.agent.tools', 'src.utils.logger'):
            self.stack.enter_context(patch(module + '.load_config', return_value=self.config))
        self.request = self.stack.enter_context(patch('src.agent.react_loop._model_request', return_value=Request('http://localhost:11434/api/chat')))

    @staticmethod
    def packet(*, calls=None, answer=None, done_reason='stop'):
        message = {'tool_calls': calls} if calls is not None else {'content': json.dumps(answer, ensure_ascii=False)}
        return BytesIO(json.dumps({'model': 'audit-response-stub', 'done': True, 'done_reason': done_reason, 'message': message, 'prompt_eval_count': 1, 'eval_count': 1}).encode())

    @staticmethod
    def call(name='calculator', args=None, identifier='audit-call'):
        return {'name': name, 'args': {'expression': '0.1+0.2'} if args is None else args, 'call_id': identifier}

    def test_open_issue_agent_01_previous_tool_evidence_survives_observation(self):
        """缺陷：第二轮Observation把第一轮结果删掉，却只附第二轮ToolMessage。"""
        context = {'observations': [{'name': 'first', 'status': 'success', 'result': {'code': 'AUDIT_PREVIOUS_EVIDENCE_731'}}, {'name': 'second', 'status': 'success', 'result': {'time': '08:00'}}]}
        decision = {'observation': '整合两个结果', 'decision': 'finish', 'task_complete': True, 'answer': '整合结果'}
        messages = [AIMessage(content='', tool_calls=[{'name': 'second', 'args': {}, 'id': 'b'}]), ToolMessage(content='{"time":"08:00"}', name='second', tool_call_id='b')]
        with patch('src.agent.react_loop.urlopen', return_value=self.packet(answer=decision)):
            observe('合并前两步事实', [], context, messages, thought={'next_step': 'tool'})
        submitted = self.request.call_args.args[0]
        self.assertIn('AUDIT_PREVIOUS_EVIDENCE_731', '\n'.join((message.content for message in submitted)))

    def test_open_issue_agent_02_suffix_filename_does_not_select_wrong_paper(self):
        """缺陷：paper.pdf被当作mypaper.pdf的匹配项，错误ID不能被当前目标纠正。"""
        invoked = []

        @tool('paper_summary')
        def capture_document(doc_id: str) -> dict:
            """只记录实际接收ID，避免模型或原文加载掩盖目标绑定结果。"""
            invoked.append(doc_id)
            return {'doc_id': doc_id}
        context = {'observations': [{'name': 'paper_list', 'status': 'success', 'result': {'papers': [{'doc_id': 'a' * 64, 'source_file': 'paper.pdf'}, {'doc_id': 'b' * 64, 'source_file': 'mypaper.pdf'}]}}]}
        wrong_call = {'function': {'name': 'paper_summary', 'arguments': {'doc_id': 'a' * 64}}}
        with patch('src.agent.react_loop.urlopen', return_value=self.packet(calls=[wrong_call])):
            list(act('总结 mypaper.pdf', {'next_step': 'tool', 'tool_name': 'paper_summary'}, [capture_document], context))
        self.assertEqual(invoked, ['b' * 64])

    def test_open_issue_agent_03_nonserializable_result_is_recorded_as_tool_error(self):
        """缺陷：输出JSON序列化在异常处理外，错误结果绕过tool_result和恢复逻辑。"""

        @tool
        def invalid_output() -> dict:
            """真实执行函数返回非有限值，模拟非法工具结果。"""
            return {'score': float('nan')}
        results = list(execute_calls([self.call('invalid_output', {})], [invalid_output]))
        self.assertEqual(results[0]['status'], 'error')
        self.assertEqual(results[0]['error_kind'], 'execution')

    def test_rewritten_search_keeps_explicit_answer_language(self):
        """真实F14/S06改写查询丢掉语言要求；检索子问题仍应保留原输出要求。"""
        received = []

        @tool('knowledge_base_search')
        def capture_query(question: str) -> dict:
            """记录实际生成与缓存使用的问题，避免只改Action提示却未传给工具。"""
            received.append(question)
            return {'answer': '语言传递探针'}
        planned = {'next_step': 'tool', 'tool_name': 'knowledge_base_search'}
        for requirement in ('Answer in English', '请用中文回答'):
            response = self.packet(calls=[{'function': {'name': 'knowledge_base_search',
                                                      'arguments': {'question': 'CaiT compute comparison'}}}])
            with patch('src.agent.react_loop.urlopen', return_value=response):
                list(act('Compare CaiT and ViT. ' + requirement + '.', planned, [capture_query]))
            self.assertIn(requirement, received[-1])
            self.assertTrue(received[-1].startswith('CaiT compute comparison'))

if __name__ == "__main__":
    unittest.main()
