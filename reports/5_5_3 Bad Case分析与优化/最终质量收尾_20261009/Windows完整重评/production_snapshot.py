"""只记录生产库逻辑内容摘要和会话数量，不导出用户对话。"""
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
from src.retrieval.vector_store import VectorStore
chunks = VectorStore().list_chunks()
rows = sorted([{'text': d.page_content, 'metadata': d.metadata} for d in chunks], key=lambda r: r['metadata'].get('chunk_id', ''))
result = {'chunks': len(rows), 'documents': len({r['metadata'].get('doc_id') for r in rows}),
          'index_content_sha256': sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
path = Path('data/sessions/memory.sqlite3').resolve()
if path.exists():
    with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as db:
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        result['session_counts'] = {name: db.execute('SELECT COUNT(*) FROM "'+name+'"').fetchone()[0] for name in tables if name in ('sessions','messages','summaries','rag_history')}
        result['session_rows_sha256'] = {name:sha256(json.dumps(sorted([list(r) for r in db.execute('SELECT * FROM "'+name+'"')],key=repr),ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest() for name in result['session_counts']}
        if len(sys.argv) > 2:
            # 仅排除本次网页核验新建的测试访客；不删除或改写任何会话记录。
            result['excluded_test_visitor'] = sys.argv[2]
            result['preexisting_session_counts'] = {}
            result['preexisting_session_rows_sha256'] = {}
            for name in result['session_counts']:
                query = ('SELECT * FROM sessions WHERE user_id != ?' if name == 'sessions' else
                         'SELECT * FROM "'+name+'" WHERE session_id IN (SELECT session_id FROM sessions WHERE user_id != ?)')
                records = sorted([list(r) for r in db.execute(query,(sys.argv[2],))],key=repr)
                result['preexisting_session_counts'][name] = len(records)
                result['preexisting_session_rows_sha256'][name] = sha256(json.dumps(records,ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest()
Path(sys.argv[1]).write_text(json.dumps(result,ensure_ascii=False,indent=2), encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
