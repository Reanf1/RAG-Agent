"""科研对话前端：真实会话存储、模拟阶段/流式事件，不启动模型。"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch
from langchain_core.messages import AIMessage


class TestStreamingFrontend(unittest.TestCase):
    def setUp(self):
        from src.utils.config import load_config
        from streamlit.testing.v1 import AppTest
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = load_config()
        for name, path in [('raw_documents', 'raw'), ('vector_index', 'index'), ('logs', 'logs'), ('session_db', 'sessions.sqlite3')]:
            self.config['paths'][name] = str(Path(self.directory.name) / path)
        for module in ['src.utils.config', 'src.utils.logger', 'src.agent.memory']:
            patcher = patch(module + '.load_config', return_value=self.config)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch('src.utils.config.check_health', return_value={'llm': {'status': 'ok'}, 'vector_database': {'status': 'ok'}})
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch('src.agent.react_loop._run_react', side_effect=lambda *a, **kw: iter(deepcopy(self.events())))
        self.core = patcher.start()
        self.addCleanup(patcher.stop)
        self.app = AppTest.from_file(str(ROOT / 'src/frontend/app.py'), default_timeout=10).run()

    @staticmethod
    def events():
        native = AIMessage(content='', tool_calls=[{'id': 'call-full-identifier', 'name': 'calculator', 'args': {}}])
        return [
            {'type': 'thought', 'iteration': 1, 'usage': {'prompt_eval_count': 10, 'eval_count': 2}, 'elapsed_seconds': .1},
            {'type': 'tool_call', 'iteration': 1, 'call_id': 'call-full-identifier', 'name': 'calculator', 'message': native,
             'usage': {'prompt_eval_count': 2, 'eval_count': 1}, 'elapsed_seconds': .1},
            {'type': 'tool_result', 'iteration': 1, 'call_id': 'call-full-identifier', 'name': 'calculator', 'status': 'success',
             'result': {'usage': {'prompt_eval_count': 5, 'eval_count': 3}}, 'elapsed_seconds': .2},
            {'type': 'observation', 'iteration': 1, 'usage': {'prompt_eval_count': 4, 'eval_count': 2}, 'elapsed_seconds': .2},
            {'type': 'token', 'answer': '逐步', 'provisional': True},
            {'type': 'token', 'answer': '逐步输出的回答。', 'provisional': True},
            {'type': 'done', 'iterations': 1, 'task_complete': True, 'stop_reason': 'task_complete', 'full_response': '逐步输出的回答。'}]

    def send(self, question):
        self.app.chat_input[0].set_value(question).run()
        self.assertFalse(self.app.exception)

    def test_three_tabs_and_inline_chat_input_replace_single_rag(self):
        self.assertEqual([t.label for t in self.app.tabs], ['科研对话', '文档检索', '知识库'])
        self.assertEqual(len(self.app.chat_input), 1)
        self.assertFalse(any(b.key == 'run_agent' for b in self.app.button))
        self.assertFalse(any(m.label in {'本次 Agent Token', 'Agent 响应耗时'} for m in self.app.metric))
        self.core.assert_not_called()

    def test_two_turns_clear_input_keep_history_and_accumulate_once(self):
        self.send('第一次计算')
        self.assertIsNone(self.app.chat_input[0].value)
        self.assertTrue(any('本次 Token：29' in c.value and 'Agent响应耗时：' in c.value for c in self.app.caption))
        self.send('继续计算')
        self.assertEqual(len(self.app.chat_message), 4)
        self.assertEqual(len(self.app.session_state['agent_messages']), 2)
        self.assertEqual(next(m.value for m in self.app.metric if m.label == '累计Token'), '58')
        self.assertEqual(next(m.value for m in self.app.metric if m.label == '对话次数'), '2')
        self.assertEqual(len(self.app.tabs[0].dataframe), 2)
        for table in self.app.tabs[0].dataframe:
            self.assertEqual(list(table.value.columns), ['阶段', '轮次', '工具', '输入Token', '输出Token', '耗时', '状态'])
        self.assertFalse(self.app.get('graphviz_chart'))
        self.app.run()
        self.assertEqual(self.core.call_count, 2)
        self.assertEqual(next(m.value for m in self.app.metric if m.label == '累计Token'), '58')

    def test_stream_exception_preserves_partial_answer_and_failed_round(self):
        def broken(*args, **kwargs):
            for event in self.events()[:-1]:
                yield event
            raise RuntimeError('模拟流中断')
        self.core.side_effect = broken
        self.send('中断样例')
        self.assertTrue(any('模拟流中断' in e.value for e in self.app.error))
        self.assertIn('逐步输出', self.app.session_state['agent_messages'][0]['answer'])
        self.assertEqual(next(m.value for m in self.app.metric if m.label == '任务完成率'), '0.0%')
        self.assertIn('失败', list(self.app.tabs[0].dataframe[0].value['状态']))
        self.app.button(key='new_conversation').click().run()
        self.assertEqual(next(m.value for m in self.app.metric if m.label == '对话次数'), '0')
        memory, user = self.app.session_state['agent_memory'], self.app.session_state['agent_user_id']
        old = next(s for s in memory.list_sessions(user) if memory.get_messages(user, s))
        self.app.button(key=f'conversation:{old}').click().run()
        self.assertTrue(any('模拟流中断' in e.value for e in self.app.error))
        self.assertEqual(len(self.app.chat_message), 2)
        self.core.assert_called_once()

    def test_unknown_usage_stays_unknown_and_new_session_resets_statistics(self):
        events = self.events()
        events[0].pop('usage')
        self.core.side_effect = lambda *a, **kw: iter(deepcopy(events))
        self.send('未知用量')
        self.assertTrue(any('本次 Token：未知' in c.value for c in self.app.caption))
        self.assertEqual(next(m.value for m in self.app.metric if m.label == '累计Token'), '未知')
        self.app.button(key='new_conversation').click().run()
        self.assertEqual(next(m.value for m in self.app.metric if m.label == '对话次数'), '0')
        self.assertFalse(self.app.chat_message)

    def test_rag_cache_is_bound_to_current_session_and_passed_to_agent(self):
        cache = self.app.session_state["agent_rag_cache"]
        self.send("缓存入口检查")
        self.assertIs(self.app.session_state["agent_rag_cache"], cache)
        tools = self.core.call_args.args[1]
        self.assertIn("knowledge_base_search", [t.name for t in tools])
        self.app.button(key="new_conversation").click().run()
        self.assertIsNot(self.app.session_state["agent_rag_cache"], cache)

    def test_agent_history_renders_separate_source_buttons_for_parallel_calls(self):
        reference = {'id': 1, 'source_file': 'attention.pdf', 'location': '第3页', 'metadata': {'page_number': 3}}
        event = {'request_id': 'saved-request', 'context': {'observations': [
            {'call_id': identifier, 'result': {'citations': [reference]}} for identifier in ('first', 'second')]}}
        self.app.session_state['agent_messages'] = [{'question': '两篇论文？', 'answer': '已保存答案',
            'complete': True, 'stop_reason': 'task_complete', 'event': event}]
        self.app.run()
        self.assertFalse(self.app.exception)
        keys = {b.key for b in self.app.button if b.label == '查看attention.pdf · 第3页'}
        self.assertEqual(keys, {'agent-citation:saved-request:first:1', 'agent-citation:saved-request:second:1'})
        self.core.assert_not_called()


class TestConversationStatistics(unittest.TestCase):
    """核验累计口径、未知历史和并行共享用量，避免UI重复记账。"""
    def snapshot(self):
        from src.utils.logger import update_agent_metrics
        metrics = {}
        for event in TestStreamingFrontend.events()[:4]:
            metrics = update_agent_metrics(metrics, event)
        metrics['response_seconds'] = 2.0
        return {'type': 'done', 'iterations': 1, 'task_complete': True, 'metrics': metrics}

    def test_totals_rates_and_unknown_legacy_samples(self):
        from src.frontend.components.trace import conversation_statistics
        first = self.snapshot()
        first['metrics']['retrievals'] = [{'status': 'success', 'returned_chunks': 5, 'seconds': .1}]
        second = deepcopy(first)
        second['metrics']['response_seconds'] = 4
        second['metrics']['tool_calls'][0]['status'] = 'error'
        second['metrics']['retrievals'] = [{'status': 'empty', 'returned_chunks': 0, 'seconds': .3}]
        messages = [{'complete': True, 'event': first}, {'complete': False, 'event': second}]
        stats = conversation_statistics(messages)
        self.assertEqual((stats['requests'], stats['tokens'], stats['seconds'], stats['mean_seconds']), (2, 58, 6, 3))
        self.assertEqual((stats['tool_successes'], stats['tool_failures'], stats['tool_rate']), (1, 1, .5))
        self.assertEqual(stats['retrieval_hit_rate'], .5)
        self.assertAlmostEqual(stats['mean_retrieval_seconds'], .2)
        messages.append({'complete': None, 'event': {}})
        unknown = conversation_statistics(messages)
        self.assertIsNone(unknown['tokens'])
        self.assertIsNone(unknown['seconds'])
        self.assertEqual((unknown['known_tokens'], unknown['completed_requests']), (58, 2))
        self.assertEqual(unknown['mean_seconds'], 3)
        self.assertEqual(conversation_statistics([])['requests'], 0)

    def test_shared_action_does_not_assign_tokens_twice_and_pending_is_visible(self):
        from src.utils.logger import update_agent_metrics
        from src.frontend.components.trace import execution_rows
        native = AIMessage(content='', tool_calls=[{'id': i, 'name': 'calculator', 'args': {}} for i in ('a', 'b')])
        metrics = {}
        for i in ('a', 'b'):
            metrics = update_agent_metrics(metrics, {'type': 'tool_call', 'iteration': 1, 'call_id': i,
                'name': 'calculator', 'message': native, 'usage': {'prompt_eval_count': 3, 'eval_count': 2}})
        pending = execution_rows({'metrics': metrics})
        self.assertEqual(sum(row['阶段'] == 'Action（共享）' for row in pending), 1)
        self.assertEqual(sum(row['状态'] == '执行中' for row in pending), 2)
        metrics = update_agent_metrics(metrics, {'type': 'tool_result', 'iteration': 1, 'call_id': 'a',
            'name': 'calculator', 'status': 'error', 'result': {}, 'elapsed_seconds': .4})
        rows = execution_rows({'type': 'done', 'iterations': 1, 'task_complete': False, 'metrics': metrics})
        failed = next(row for row in rows if row['阶段'] == '工具内部')
        self.assertEqual((failed['状态'], failed['耗时'], failed['输入Token']), ('失败', '0.400 秒', '未知'))
        self.assertEqual(rows[-1]['阶段'], '任务结束')
        self.assertEqual(rows[-1]['状态'], '失败')
        self.assertNotIn('执行中', [row['状态'] for row in rows])
        self.assertEqual(metrics['tokens']['known_total'], 5)

    def test_skipped_action_precedes_observation(self):
        from src.utils.logger import update_agent_metrics
        from src.frontend.components.trace import execution_rows
        metrics = {}
        for event in [{'type': 'thought', 'iteration': 1}, {'type': 'action_skipped', 'iteration': 1},
                      {'type': 'observation', 'iteration': 1}]:
            metrics = update_agent_metrics(metrics, event)
        self.assertEqual([row['阶段'] for row in execution_rows({'metrics': metrics})],
                         ['Thought', 'Action（跳过）', 'Observation'])


if __name__ == '__main__':
    unittest.main()
