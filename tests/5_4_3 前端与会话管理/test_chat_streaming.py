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
from src.agent.react_loop import _run_react


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

    def test_agent_history_renders_source_label_for_each_call(self):
        """历史回放按工具调用分别展示引用来源文字标签，不重新执行模型或工具。"""
        reference = {'id': 1, 'source_file': 'attention.pdf', 'location': '第3页', 'metadata': {'page_number': 3}}
        event = {'request_id': 'saved-request', 'context': {'observations': [
            {'call_id': identifier, 'result': {'citations': [reference]}} for identifier in ('first', 'second')]}}
        self.app.session_state['agent_messages'] = [{'question': '两篇论文？', 'answer': '已保存答案',
            'complete': True, 'stop_reason': 'task_complete', 'event': event}]
        self.app.run()
        self.assertFalse(self.app.exception)
        labels = [c.value for c in self.app.caption]
        self.assertEqual(labels.count('attention.pdf · 第3页'), 2)
        self.core.assert_not_called()

    def test_same_document_page_keeps_one_source_label_per_answer(self):
        """同一轮同页多块引用只展示一条来源标签，不重复堆叠。"""
        def reference(identifier, number):
            return {'id': number, 'source_file': 'ViT.pdf', 'location': '第21页',
                    'metadata': {'doc_id': identifier, 'page_number': 21}}
        event = {'request_id': 'same-page', 'context': {'observations': [
            {'call_id': 'first', 'result': {'citations': [reference('a' * 64, 1), reference('a' * 64, 2)]}},
            {'call_id': 'second', 'result': {'citations': [reference('b' * 64, 2)]}}]}}
        self.app.session_state['agent_messages'] = [{'question': '位置编码？', 'answer': '已保存答案',
            'complete': True, 'stop_reason': 'task_complete', 'event': event}]
        self.app.run()
        self.assertFalse(self.app.exception)
        labels = [c.value for c in self.app.caption]
        self.assertEqual(labels.count('ViT.pdf · 第21页'), 3)

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

    def test_low_candidate_confirm_and_cancel_do_not_rely_on_model_permission(self):
        self.pending_candidate()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("需人工核对的候选原文" in m.value for m in self.app.markdown))
        self.app.button(key="cancel-rag:pending-one").click().run()
        self.core.assert_not_called()
        self.assertFalse(self.app.session_state["agent_pending_rag"])
        self.pending_candidate()
        self.app.button(key="confirm-rag:pending-one").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.core.call_count, 1, [e.value for e in self.app.error])
        self.assertEqual(self.core.call_args.args[2]["confirmed_rag_args"], {"question": "ViT输入？", "doc_id": None})
        self.assertFalse(self.app.session_state["agent_pending_rag"])
        self.app.run()
        self.core.assert_called_once()

    def test_session_switch_invalidates_pending_candidates(self):
        self.pending_candidate()
        self.app.button(key="new_conversation").click().run()
        self.assertFalse(self.app.session_state["agent_pending_rag"])
        self.core.assert_not_called()

    def test_comparison_confirmation_runs_real_tool_and_agent_completion(self):
        """页面确认贯通真实工具、Agent与会话；仅模型HTTP和索引版本为受控样例。"""
        from io import BytesIO
        import json
        from src.data_loader import batch_import, create_import_tasks
        from src.generation.rag_pipeline import prepare_rag_context

        tasks = create_import_tasks([
            ("论文A.md", b"Model A uses method A and achieves accuracy 85% on Dataset X."),
            ("论文B.md", b"Model B uses method B and achieves accuracy 90% on Dataset Y."),
        ])
        list(batch_import(tasks, self.config["paths"]["raw_documents"]))
        documents = [task["documents"][0] for task in tasks]
        for index, doc in enumerate(documents):
            doc.metadata["chunk_id"] = f"confirmation-{index}"
        identifiers = [doc.metadata["doc_id"] for doc in documents]
        question = "对比两篇论文的方法、数据集与实验结果。"
        args = dict(zip(("paper_a_id", "paper_b_id"), identifiers))
        self.core.side_effect = _run_react  # 恢复产品执行流程，不能以预设done事件证明完成。

        def packet(request, timeout):
            properties = json.loads(request.data)["format"]["properties"]
            choice = {key: {"text": "本篇的" + key.split("_")[0],
                            "reference_id": value["properties"]["reference_id"]["enum"][0]}
                      for key, value in properties.items()}
            return BytesIO(json.dumps({"model": "protocol-fixture", "done": True, "done_reason": "stop",
                "prompt_eval_count": 30, "eval_count": 10, "message": {"content": json.dumps(choice)}}).encode())

        with patch("src.agent.tools.load_config", return_value=self.config), \
                patch("src.generation.rag_pipeline.load_config", return_value=self.config), \
                patch("src.generation.cache.cache_scope", return_value="scope-v1"), \
                patch("src.retrieval.vector_store.VectorStore"), \
                patch("src.retrieval.hybrid_retriever.HybridRetriever") as retriever, \
                patch("src.generation.rag_pipeline.urlopen", side_effect=packet) as http, \
                patch("src.agent.react_loop.urlopen", side_effect=AssertionError("确认后无需重新规划或改写报告")):
            context = prepare_rag_context(question, list(zip(documents, (.01, .9))))
            papers = [{"label": label, "doc_id": identifier, "truncated": False,
                       "references": [ref for ref in context["references"] if ref["metadata"]["doc_id"] == identifier]}
                      for label, identifier in zip(("A", "B"), identifiers)]
            base = {"papers": papers, "missing_dimensions": [], "low_relevance_dimensions": ["论文A：方法"],
                    "low_relevance_papers": []}
            approval = {"tool_name": "paper_compare", "question": question, "args": args,
                        "session_id": self.app.session_state["agent_session_id"], "scope": "scope-v1",
                        "context": context, "base": base}
            original = deepcopy(approval)
            event = {"request_id": "compare-review", "context": {"observations": [{"name": "paper_compare",
                "status": "success", "call_id": "compare-call", "result": {"status": "needs_confirmation",
                "references": context["references"], "citations": [], "confirmation_id": "compare-pending"}}]}}
            self.app.session_state["agent_messages"] = [{"question": question, "answer": "待确认", "complete": False, "event": event}]
            self.app.session_state["agent_pending_rag"] = {"compare-pending": deepcopy(approval)}
            self.app.run()
            self.app.button(key="cancel-rag:compare-pending").click().run()
            self.assertFalse(self.app.exception)
            self.core.assert_not_called()
            http.assert_not_called()
            self.app.session_state["agent_pending_rag"] = {"compare-pending": deepcopy(approval)}
            self.app.run()
            self.app.button(key="confirm-rag:compare-pending").click().run()
            self.assertFalse(self.app.exception)
            message = self.app.session_state["agent_messages"][-1]
            self.assertTrue(message["complete"], message)
            self.assertEqual(self.core.call_count, 1)
            self.assertEqual(self.core.call_args.args[2]["confirmed_rag_args"], args)
            self.assertEqual(http.call_count, 1)
            retriever.assert_not_called()
            observation = message["event"]["context"]["observations"][-1]["result"]
            self.assertTrue(observation["confirmed"])
            self.assertEqual({ref["metadata"]["doc_id"] for ref in observation["citations"]}, set(identifiers))
            self.assertEqual([row["a"]["text"] for row in observation["comparison"]], [documents[0].page_content] * 3)
            self.assertEqual(approval, original)
            self.assertFalse(self.app.session_state["agent_pending_rag"])
            self.app.run()
            self.assertEqual(http.call_count, 1)
            self.assertEqual(self.core.call_count, 1)


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

    def test_cache_and_confirmation_reuse_are_not_new_retrieval_samples(self):
        from src.frontend.components.trace import conversation_statistics
        event = self.snapshot()
        event["metrics"]["retrievals"] = [{"status": "success", "returned_chunks": 5, "seconds": .3},
                                         {"status": "cache", "returned_chunks": 0, "seconds": 0},
                                         {"status": "confirmed", "returned_chunks": 0, "seconds": 0}]
        stats = conversation_statistics([{"complete": True, "event": event}])
        self.assertEqual(stats["retrieval_count"], 1)
        self.assertEqual(stats["retrieval_hit_rate"], 1)
        self.assertEqual(stats["mean_retrieval_seconds"], .3)


if __name__ == '__main__':
    unittest.main()
