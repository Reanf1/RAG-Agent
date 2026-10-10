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


    def pending_candidate(self):
        # 明确样例，不作真实模型测试；按钮必须只处理当前会话的候选。
        reference = {"id": 1, "source_file": "review.md", "location": "行1", "text": "需人工核对的候选原文", "score": .01, "metadata": {}}
        event = {"request_id": "low-request", "context": {"observations": [{"call_id": "low-call", "result": {
            "status": "needs_confirmation", "answer": "待确认", "citations": [], "references": [reference],
            "confirmation_id": "pending-one"}}]}}
        self.app.session_state["agent_pending_rag"] = {"pending-one": {"question": "论文输入？", "tool_question": "ViT输入？",
            "doc_id": None, "session_id": self.app.session_state["agent_session_id"]}}
        self.app.session_state["agent_messages"] = [{"question": "论文输入？", "answer": "待确认", "complete": False, "event": event}]
        self.app.run()


if __name__ == '__main__':
    unittest.main()
