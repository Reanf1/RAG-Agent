"""独立审计缺陷回归：移植原预期行为断言；模型报文和微型向量为明确替身。"""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.agent.memory import MemoryManager
from src.generation.cache import SemanticCache

class ExtendedAgentBoundaryAudit(unittest.TestCase):

    def test_same_user_sessions_remain_isolated_after_reopen_and_clear(self):
        """真实SQLite，只替代Token计数；不同会话不得互读或一起清空。"""
        with tempfile.TemporaryDirectory(prefix='same-user-memory-') as directory:
            database = Path(directory) / 'memory.sqlite3'
            manager = MemoryManager(database)
            (first, second) = (manager.create_session('同一用户'), manager.create_session('同一用户'))
            manager.append_turn('同一用户', first, '标识A_1723', '已记住A_1723')
            manager.append_turn('同一用户', second, '标识B_9481', '已记住B_9481')
            reopened = MemoryManager(database)
            with patch('src.agent.memory.count_history_tokens', return_value=0):
                context = reopened.get_context('同一用户', second)
            self.assertEqual([row['content'] for row in context['history']], ['标识B_9481', '已记住B_9481'])
            reopened.clear_session('同一用户', first)
            self.assertEqual(reopened.get_messages('同一用户', first), [])
            self.assertEqual([row.content for row in reopened.get_messages('同一用户', second)], ['标识B_9481', '已记住B_9481'])

    def lookup_changed_condition(self, original, changed):
        """向量用明确高相似替身，回归保护规则；真实余弦另存cache_constraints。"""
        result = {'type': 'done', 'done_reason': 'stop', 'generation_mode': 'grounded', 'answer': '只适用于原条件的合成夹具答案', 'citations': [{'id': 1}], 'usage': {}}
        with patch('src.generation.cache.get_embeddings') as embeddings:
            embeddings.return_value.embed_query.return_value = [1.0, 0.0]
            cache = SemanticCache()
            self.assertTrue(cache.put(original, result, 'isolated-fixture'))
            return cache.lookup(changed, 'isolated-fixture')

    def test_arabic_numbers_block_even_high_similarity(self):
        self.assertIsNone(self.lookup_changed_condition('100个样本时准确率是多少？', '200个样本时准确率是多少？'))

    def test_chinese_number_change_does_not_reuse_old_answer(self):
        self.assertIsNone(self.lookup_changed_condition('在一百个训练样本的设置下，实验的准确率是多少？', '在二百个训练样本的设置下，实验的准确率是多少？'))

    def test_changed_negation_target_does_not_reuse_old_answer(self):
        self.assertIsNone(self.lookup_changed_condition('哪些模型没有使用有标签数据？', '哪些模型没有使用无标签数据？'))

if __name__ == "__main__":
    unittest.main()
