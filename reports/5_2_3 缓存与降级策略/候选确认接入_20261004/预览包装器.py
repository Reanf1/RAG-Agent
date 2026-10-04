"""仅供Mac界面核验：真实临时原文/Chroma/SQLite，模型、向量和检索为明确样例。"""
from contextlib import ExitStack
from io import BytesIO
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import streamlit as st
from src.utils.config import load_config
from tests.helpers import SmallEmbeddings

preview = Path('/private/tmp/rag-confirm-preview-20261004')
config = load_config()
for name in ('raw_documents', 'vector_index', 'logs'):
    config['paths'][name] = str(preview / name)
config['paths']['session_db'] = str(preview / 'sessions.sqlite3')
embedding = SmallEmbeddings()
modules = ('src.utils.config', 'src.utils.logger', 'src.agent.tools', 'src.agent.memory',
           'src.agent.react_loop', 'src.agent.router', 'src.generation.cache', 'src.generation.rag_pipeline',
           'src.retrieval.vector_store', 'src.retrieval.hybrid_retriever', 'src.chunking')

with ExitStack() as stack:
    for module in modules:
        stack.enter_context(patch(module + '.load_config', return_value=config))
    for module in ('src.retrieval.vector_store', 'src.generation.cache'):
        stack.enter_context(patch(module + '.get_embeddings', return_value=embedding))
    stack.enter_context(patch('src.utils.config.check_health', return_value={
        'llm': {'status': 'ok', 'model': '模拟模型（界面核验）'}, 'vector_database': {'status': 'ok'}}))
    from src.data_loader import create_import_tasks
    from src.retrieval.vector_store import batch_build_index, VectorStore
    if not (preview / 'raw_documents').exists():
        tasks = create_import_tasks([('界面验证论文.md', '# 明确构造的界面验证样例\n\nViT使用图像patch作为输入。本文件只用于检查候选原文、引用和确认按钮。'.encode())])
        list(batch_build_index(tasks, preview / 'raw_documents'))
    document = VectorStore().list_chunks()[0]
    if 'preview_counts' not in st.session_state:
        st.session_state.preview_counts = {'retrieval': 0, 'generation': 0}
    def agent_response(request, **kwargs):
        payload = json.loads(request.data)
        query = json.loads(payload['messages'][1]['content'])
        if payload.get('tools'):
            message = {'tool_calls': [{'function': {'name': 'knowledge_base_search', 'arguments': {'question': query['question']}}}]}
        else:
            answer = query['context']['observations'][-1]['result']['answer']
            message = {'content': json.dumps({'observation': '保留模拟工具答案。', 'decision': 'finish',
                                             'task_complete': True, 'answer': answer}, ensure_ascii=False)}
        packet = {'model': 'mock', 'done': True, 'done_reason': 'stop', 'prompt_eval_count': 20, 'eval_count': 5, 'message': message}
        return BytesIO(json.dumps(packet, ensure_ascii=False).encode() + (b'\n' if payload.get('stream') else b''))
    # 工具在线程中执行，不访问线程外的Streamlit状态。
    counts = st.session_state.preview_counts
    def retrieve(question, **kwargs):
        counts['retrieval'] += 1
        return [(document, .01 if '低相关' in question else .9)]
    def rag_response(*args, **kwargs):
        counts['generation'] += 1
        return BytesIO(json.dumps({'model': 'mock', 'done': True, 'done_reason': 'stop',
            'prompt_eval_count': 80, 'eval_count': 10,
            'message': {'content': '界面验证样例：ViT使用图像patch作为输入。[参考文档1]'}}).encode())
    stack.enter_context(patch('src.retrieval.hybrid_retriever.HybridRetriever.search', side_effect=retrieve))
    stack.enter_context(patch('src.generation.rag_pipeline.urlopen', side_effect=rag_response))
    stack.enter_context(patch('src.agent.react_loop.urlopen', side_effect=agent_response))
    stack.enter_context(patch('src.agent.react_loop.think', return_value={
        'type': 'thought', 'thought': '模拟规划：调用核心RAG工具。', 'next_step': 'tool',
        'tool_name': 'knowledge_base_search', 'parallel_tools': [], 'model': 'mock',
        'usage': {'prompt_eval_count': 0, 'eval_count': 0}}))
    runpy.run_path(str(ROOT / 'src/frontend/app.py'), run_name='__main__')
    st.caption(f"界面模拟核验计数：检索 {counts['retrieval']} 次，生成 {counts['generation']} 次。时间和Token均不是实际模型性能。")
