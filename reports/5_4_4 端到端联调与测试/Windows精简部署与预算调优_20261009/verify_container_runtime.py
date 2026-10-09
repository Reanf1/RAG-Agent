"""在独立容器中核验真实SQLite、Chroma重启及本地模型；不访问生产数据。"""

import json
from pathlib import Path
import socket
import sqlite3
import sys
from time import perf_counter

sys.path.insert(0, '/app')
import torch
torch.set_num_threads(2)
from langchain_core.documents import Document
from src.agent.memory import MemoryManager
from src.generation.rag_pipeline import build_context, generate_answer
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore

started = perf_counter()
directory = Path('/app/data/simplify-runtime')
directory.mkdir(parents=True, exist_ok=True)
phase = sys.argv[1]
store = VectorStore(directory / 'index')
memory = MemoryManager(directory / 'sessions.sqlite3')
if phase == 'seed':
    assert store.count() == 0, '只允许初始化本轮新建索引'
    added = store.add_chunks([Document(page_content='容器本地复测代号是 SIMPLIFYDOCKER20261009。', metadata={
        'doc_id': 'a' * 64, 'chunk_id': 'container-probe', 'source_file': '容器探针.txt', 'line_start': 1, 'line_end': 1})])
    session = memory.create_session('audit')
    memory.append_turn('audit', session, '容器探针', '已记录')
    (directory / 'session.json').write_text(json.dumps({'id': session}), encoding='utf-8')
    result = {'added_chunks': added, 'stored_chunks': store.count(), 'sqlite_version': sqlite3.sqlite_version,
              'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available()}
else:
    assert store.count() == 1
    session = json.loads((directory / 'session.json').read_text(encoding='utf-8'))['id']
    assert memory.get_messages('audit', session)[-1].content == '已记录'
    question = '容器本地复测代号是什么？'
    ranked = Reranker().rerank(question, store.search(question), k=1)
    answer = generate_answer(question, build_context(question, ranked))
    assert 'SIMPLIFYDOCKER20261009' in answer['answer'] and answer['citations'] and not answer['warnings'], answer
    # 使用公共IP直接探测，不把DNS失败等同于完整离线隔离证据。
    blocked = False
    try:
        with socket.create_connection(('1.1.1.1', 443), timeout=3):
            pass
    except OSError:
        blocked = True
    assert blocked, '业务容器不应连接公网'
    result = {'after_new_process_chunks': store.count(), 'session_restored': True, 'outbound_blocked': blocked,
              'rerank_score': ranked[0][1], 'generation': answer}
print(json.dumps({'phase': phase, 'seconds': perf_counter()-started, 'result': result,
                  'boundary': '真实容器与本地M3E/BGE/Qwen，探针文字为合成数据；不代表论文质量或总体性能'}, ensure_ascii=False, indent=2))
