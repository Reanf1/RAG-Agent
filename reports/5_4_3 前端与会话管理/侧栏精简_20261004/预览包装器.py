"""侧栏布局临时核验：仅使用临时原文/索引/会话及小型测试向量，不连接LLM。"""
from pathlib import Path
import runpy
import sys
from contextlib import ExitStack
from unittest.mock import patch

ROOT = Path('/Users/rean/github/RAG+Agent')
sys.path.insert(0, str(ROOT))
from src.utils.config import load_config
from tests.helpers import SmallEmbeddings

config = load_config()
preview = Path('/private/tmp/rag-sidebar-preview-20261004')
for key, name in [('raw_documents', 'raw'), ('vector_index', 'index'), ('logs', 'logs')]:
    config['paths'][key] = str(preview / name)
config['paths']['session_db'] = str(preview / 'memory.sqlite3')
modules = ['src.utils.config', 'src.agent.memory', 'src.utils.logger', 'src.agent.react_loop',
           'src.agent.tools', 'src.agent.router', 'src.retrieval.vector_store',
           'src.retrieval.hybrid_retriever', 'src.retrieval.reranker',
           'src.generation.rag_pipeline', 'src.generation.cache', 'src.chunking']
with ExitStack() as stack:
    for name in modules:
        stack.enter_context(patch(name + '.load_config', return_value=config))
    stack.enter_context(patch('src.retrieval.vector_store.get_embeddings', return_value=SmallEmbeddings()))
    stack.enter_context(patch('src.generation.rag_pipeline.urlopen', side_effect=RuntimeError('布局核验不连接模型')))
    from src.agent.memory import MemoryManager
    from src.data_loader import create_import_tasks
    from src.retrieval.vector_store import batch_build_index
    if not (preview / 'raw').exists():
        tasks = create_import_tasks([('前端验收说明.txt', '仅用于侧栏布局核验。'.encode())])
        list(batch_build_index(tasks, preview / 'raw'))
    memory = MemoryManager(preview / 'memory.sqlite3')
    user = '1' * 32
    if not memory.list_sessions(user):
        for title in ['分析ViT论文的方法', '对比论文实验结果']:
            identifier = memory.create_session(user)
            memory.append_turn(user, identifier, title, '临时测试会话，仅用于核验页面布局与切换。')
    runpy.run_path(str(ROOT / 'src/frontend/app.py'), run_name='__main__')
