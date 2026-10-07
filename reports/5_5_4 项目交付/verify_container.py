"""在本次独立容器数据卷验证真实导入、索引、Agent和持久化，禁止用于用户知识库。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import socket
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reopen', action='store_true')
    parser.add_argument('--reuse-index', action='store_true', help='证据脚本修正后复用已验收的独立索引，不重复导入')
    args = parser.parse_args()
    if not Path('/.dockerenv').is_file(): raise RuntimeError('本脚本只用于独立验收容器')
    import torch
    import sqlite3
    torch.set_num_threads(4)
    from src.utils.config import load_config, check_health
    from src.retrieval.vector_store import VectorStore
    from src.agent.memory import MemoryManager, run_session
    config = load_config()
    if args.reopen:
        report = json.loads(args.output.read_text())
        assert VectorStore().count() == report['chunks']
        memory = MemoryManager()
        assert memory.get_messages(report['user_id'], report['session_id'])[-1].content == report['agent']['full_response']
        report['new_process_reopen'] = {'chunks': VectorStore().count(), 'history_messages': len(memory.get_messages(report['user_id'], report['session_id']))}
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print('新进程读取索引与会话通过', flush=True)
        return
    from src.data_loader import create_import_tasks, batch_import
    from src.chunking import split_documents
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.frontend.components.documents import read_pdf_page
    if args.output.exists() or (Path(config['paths']['vector_index']).exists() and not args.reuse_index): raise FileExistsError('必须使用独立的新数据卷，或明确复用独立验收索引')
    report = {'status': 'running', 'started_at': datetime.now().astimezone().isoformat(), 'python': platform.python_version(), 'sqlite': sqlite3.sqlite_version, 'platform': platform.platform(), 'config': config}
    # Agent事件包含LangChain消息，保存其真实字段以便复核，不能用对象字符串代替。
    def save(): args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=lambda value: value.model_dump())+'\n', encoding='utf-8')
    save()
    assert platform.python_version() == '3.10.10'
    try:
        with socket.create_connection(('1.1.1.1', 443), timeout=3):
            raise AssertionError('业务容器能够访问外网，隔离失败')
    except OSError as error: report['external_tcp_blocked'] = f'{type(error).__name__}: {error}'
    report['health_before'] = check_health()
    assert report['health_before']['llm']['status'] == 'ok'
    report['source_sha256'] = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    if args.reuse_index:
        store = VectorStore()
        report.update(chunks=store.count(), reused_acceptance_index=True)
        assert report['chunks'] == 167
    else:
        assert report['health_before']['vector_database']['status'] == 'not_initialized'
        tasks = create_import_tasks([(args.pdf.name, args.pdf.read_bytes())])
        started = perf_counter()
        report['import_progress'] = list(batch_import(tasks, config['paths']['raw_documents']))
        assert tasks[0]['status'] == 'success'
        chunks = split_documents(tasks[0]['documents'])
        store = VectorStore()
        assert store.add_chunks(chunks) == len(chunks)
        report.update(chunks=len(chunks), import_index_seconds=perf_counter()-started, repeat_added=store.add_chunks(chunks))
        assert report['repeat_added'] == 0
    report['health_after'] = check_health()
    assert report['health_after']['status'] == 'ok'
    save();print(f"独立验收索引 {report['chunks']} 块可用；复用={args.reuse_index}", flush=True)
    # 离线验收使用论文同语言问题；跨语言排名按用户要求另行暂缓。
    started=perf_counter();hits=HybridRetriever(store).search('Which three datasets were used to pretrain Vision Transformer models?', k=5, rerank=True)
    assert hits and all(d.metadata['doc_id'] == report['source_sha256'] for d, _ in hits)
    report['retrieval'] = {'seconds': perf_counter()-started, 'top5': [{'text': d.page_content, 'metadata': d.metadata, 'score': score} for d, score in hits]}
    meta=hits[0][0].metadata
    page=read_pdf_page(Path(config['paths']['raw_documents']), {'metadata':meta,'source_file':meta['source_file']})
    report['original_page'] = {'page': page['page_number'], 'filename': page['filename'], 'png_bytes': len(page['image']), 'pdf_bytes': len(page['pdf'])}
    save();print('混合重排与物理页读取通过', flush=True)
    memory=MemoryManager();report['user_id']='container-verification';report['session_id']=memory.create_session(report['user_id'])
    save()
    question=f"Use knowledge_base_search to answer: which three datasets were used to pretrain Vision Transformer models? doc_id={report['source_sha256']}"
    started=perf_counter();events=list(run_session(question, report['user_id'], report['session_id'], memory=memory, stream=True))
    done=events[-1];report['agent']={**done, 'seconds': perf_counter()-started, 'stream_token_events': sum(e['type']=='token' for e in events)}
    report['agent_events']=events
    save()  # 完成判断失败时也保留真实模型正文和轨迹，不能只留下成功样例。
    assert done['type']=='done' and done['task_complete'], done
    assert any(e['type']=='tool_call' and e['name']=='knowledge_base_search' for e in events)
    assert any(e['type']=='tool_result' and e['name']=='knowledge_base_search' and e['status']=='success' and e['result']['citations'] for e in events)
    assert memory.get_messages(report['user_id'],report['session_id'])[-1].content==done['full_response']
    report.update(status='passed',completed_at=datetime.now().astimezone().isoformat());save()
    print('实际Agent知识库回答、引用、流式和会话写入通过',flush=True)


if __name__=='__main__': main()
