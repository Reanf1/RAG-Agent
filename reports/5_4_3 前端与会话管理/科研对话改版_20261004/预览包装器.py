"""科研对话临时核验：仅使用临时原文/索引/会话及小型测试向量，不连接LLM。"""
from pathlib import Path
import runpy
import sys
import time
import importlib.util
from contextlib import ExitStack
from unittest.mock import patch

ROOT = Path('/Users/rean/github/RAG+Agent')
sys.path.insert(0, str(ROOT))
from src.utils.config import load_config
from tests.helpers import SmallEmbeddings

config = load_config()
preview = Path('/private/tmp/rag-chat-preview-20261004')
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
    # 状态为明确的布局样例，不代表真实LLM可用性。
    stack.enter_context(patch('src.utils.config.check_health', return_value={
        'llm': {'status': 'ok', 'model': config['llm']['model']},
        'vector_database': {'status': 'ok'}}))
    # 延时模拟公开事件，仅用于验证转圈和逐步更新，不作模型性能数据。
    spec = importlib.util.spec_from_file_location('chat_fixture', ROOT / 'tests/5_4_3 前端与会话管理/test_chat_streaming.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    def preview_events(*args, **kwargs):
        for event in fixture.TestStreamingFrontend.events():
            time.sleep(2)
            if event['type'] == 'token':
                event['answer'] = '这是界面核验样例，正在逐步显示。' if len(event['answer']) < 5 else '这是界面核验样例，正在逐步显示。\n\n每轮回答分别展示 Token 与响应耗时，真实回答由本地 Agent 生成。'
            if event['type'] == 'done':
                event['full_response'] = '这是界面核验样例，正在逐步显示。\n\n每轮回答分别展示 Token 与响应耗时，真实回答由本地 Agent 生成。'
            yield event
    stack.enter_context(patch('src.agent.react_loop._run_react', side_effect=preview_events))
    from src.frontend.components.documents import delete_document
    from src.agent.memory import MemoryManager
    from src.data_loader import create_import_tasks
    from src.retrieval.vector_store import batch_build_index
    if not (preview / 'raw').exists():
        tasks = create_import_tasks([('文档A_界面验证.md', '# 文档 A\n\n这是知识库界面测试样例，用于核验文件选择与全文展示。\n\n## 内容检查\n\n| 检查项目 | 预期 |\n| --- | --- |\n| 文件选择 | 右侧显示对应内容 |\n| 内容展示 | 保留段落与表格 |'.encode()),
                                     ('文档B_切换验证.txt', '文档 B\n\n这是另一份界面测试文件。\n选择它后，右侧应显示这份原文，并替换文档 A 的内容。'.encode()),
                                     ('已归档论文.txt', '仅用于知识归档恢复核验。'.encode())])
        list(batch_build_index(tasks, preview / 'raw'))
        from hashlib import sha256
        delete_document(preview / 'raw', preview / 'index', sha256(tasks[2]['data']).hexdigest())
    memory = MemoryManager(preview / 'memory.sqlite3')
    user = '1' * 32
    if not memory.list_sessions(user) and not memory.list_sessions(user, archived=True):
        for title in ['分析ViT论文的方法', '对比论文实验结果']:
            identifier = memory.create_session(user)
            memory.append_turn(user, identifier, title, '临时测试会话，仅用于核验页面布局与切换。\n\n' * 20)
        identifier = memory.create_session(user)
        memory.append_turn(user, identifier, '待恢复的科研会话', '归档会话布局核验样例。')
        memory.delete_session(user, identifier)
    runpy.run_path(str(ROOT / 'src/frontend/app.py'), run_name='__main__')
