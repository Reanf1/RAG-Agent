"""只读核验原会话未丢失、原文及当前索引；仅导出本轮测试会话的消息。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project', type=Path, required=True)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--phase', required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.project))
from src.retrieval.vector_store import VectorStore

old = sqlite3.connect(str(args.run / 'original-sessions' / 'memory.sqlite3'))
current = sqlite3.connect(str(args.project / 'data/sessions/memory.sqlite3'))
checks = []
for table in ['sessions', 'messages', 'summaries', 'rag_history']:
    # 只保存数量和集合包含关系，不导出其他用户的问答内容。
    before = set(old.execute('SELECT * FROM ' + table).fetchall())
    after = set(current.execute('SELECT * FROM ' + table).fetchall())
    checks.append({'table': table, 'before_rows': len(before), 'after_rows': len(after),
                   'all_original_rows_preserved': before.issubset(after)})
visitor = '70d3fd048a7a45c29a5bd7a436423824'
sessions = [r[0] for r in current.execute('SELECT session_id FROM sessions WHERE user_id=?', (visitor,))]
messages = [{'session_id': session, 'messages': [dict(role=r[0], content=r[1], details=json.loads(r[2]))
              for r in current.execute('SELECT role,content,details FROM messages WHERE session_id=? ORDER BY id', (session,))]}
            for session in sessions]
old.close(); current.close()
store = VectorStore(args.project / 'data/index')
chunks = store.list_chunks()
expected = json.loads((args.run / 'reparse-build.json').read_text(encoding='utf-8'))
files = []
for row in expected['documents']:
    path = args.project / 'data/raw' / row['doc_id'] / row['file']
    status = json.loads((path.parent / '.index_status.json').read_text(encoding='utf-8'))
    actual_ids = {d.metadata['chunk_id'] for d in chunks if d.metadata['doc_id'] == row['doc_id']}
    files.append({'file': row['file'], 'original_sha256_unchanged': hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],
                  'chunks': len(actual_ids), 'expected_ids_match': actual_ids==set(row['chunk_ids']), 'index_status': status})
passed = all(r['all_original_rows_preserved'] for r in checks) and len(chunks)==276 and all(r['original_sha256_unchanged'] and r['expected_ids_match'] for r in files)
result = {'checked_at':datetime.now().astimezone().isoformat(),'phase':args.phase,'original_sessions':checks,
          'audit_sessions':messages,'active_chunks':len(chunks),'documents':files,'passed':passed}
with (args.run / ('native-state-'+args.phase+'.json')).open('x',encoding='utf-8') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2)
print(json.dumps({'phase':args.phase,'active_chunks':len(chunks),'original_sessions':checks,'passed':passed},ensure_ascii=False))
raise SystemExit(0 if passed else 1)
