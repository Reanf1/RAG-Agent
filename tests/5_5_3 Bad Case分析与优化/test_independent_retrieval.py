"""独立审计缺陷回归：移植原预期行为断言；模型报文和微型向量为明确替身。"""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.data_loader import create_import_tasks
from src.frontend.components.documents import list_documents
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore, batch_build_index
from src.utils.config import load_config

class AuditEmbeddings(Embeddings):
    """原审计的明确字符计数向量，只核验真实Chroma写入流程。"""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(text.count('农')), float(text.count('模')), float(len(text) % 7), 1.0]

class RetrievalAudit(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rag-retrieval-audit-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def store(self):
        return VectorStore(self.root / 'index', AuditEmbeddings())

    def test_reindex_after_chunk_config_change_should_not_mix_old_strategy(self):
        """已知缺陷 R1：再次导入新策略后旧块仍在，成功状态与列表冲突。"""
        raw = self.root / 'raw'
        store = self.store()
        config = deepcopy(load_config())
        config['chunking'] = {'strategy': 'fixed', 'chunk_size': 20, 'chunk_overlap': 0}
        data = ('第一句的科研内容。第二句的实验结果。' * 8).encode()
        first = create_import_tasks([('论文.txt', data)])
        with patch('src.chunking.load_config', return_value=config):
            list(batch_build_index(first, raw, vector_store=store))
        self.assertEqual(first[0]['status'], 'success')
        original_count = store.count()
        manifest = next(raw.glob('*/.index_count'))
        original_manifest = manifest.read_bytes()
        config['chunking']['strategy'] = 'semantic'
        second = create_import_tasks([('论文.txt', data)])
        with patch('src.chunking.load_config', return_value=config):
            list(batch_build_index(second, raw, vector_store=store))
        self.assertEqual(second[0]['status'], 'failed')
        self.assertIn('新索引目录重建', second[0]['error'])
        third = create_import_tasks([('论文.txt', data)])
        with patch('src.chunking.load_config', return_value=config):
            list(batch_build_index(third, raw, vector_store=store))
        row = list_documents(raw, self.root / 'index')[0]
        observed = {'first_chunk_count': first[0]['chunk_count'], 'second_chunk_count': second[0]['chunk_count'], 'second_task_status': second[0]['status'], 'third_task_status': third[0]['status'], 'third_added_chunks': third[0]['added_chunks'], 'actual_chunks': store.count(), 'list_status': row['index_status'], 'strategies': sorted({chunk.metadata['chunk_strategy'] for chunk in store.list_chunks()})}
        self.assertEqual(row['index_status'], '已向量化', json.dumps(observed, ensure_ascii=False))
        self.assertEqual(third[0]['status'], 'failed')
        self.assertEqual(store.count(), original_count)
        self.assertEqual(observed['strategies'], ['fixed'])
        self.assertEqual(manifest.read_bytes(), original_manifest)

    def test_hybrid_should_refill_after_discarding_twenty_image_placeholders(self):
        """候选补位边界 R2：用排名桩证明过滤后不补取，不能据此推断真实召回率。"""
        body = Document(page_content='采用梯度下降优化损失函数', metadata={'chunk_id': 'body', 'doc_id': 'a'})
        placeholders = [Document(page_content=f'[图像区域 {i}：原文第 {i + 1} 页]', metadata={'chunk_id': f'image-{i}', 'doc_id': 'a'}) for i in range(20)]
        store = Mock(top_k=5)
        store.search.side_effect = lambda query, k, doc_id: [(doc, 1.0) for doc in (placeholders + [body])[:k]]
        store.list_chunks.return_value = placeholders + [body]
        found = HybridRetriever(store).search('图像区域的方法', rerank=False)
        self.assertIn('body', [doc.metadata['chunk_id'] for (doc, _) in found], '21个候选中前20个为图片占位；过滤后返回0个，未补取第21个正文')

if __name__ == "__main__":
    unittest.main()
