"""Windows集中验收：独立测试数据，并以行摘要核对原资料和会话。"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from datetime import datetime

mode, output = sys.argv[1], Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
root = Path.cwd()

def save(name, value):
    (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def snapshot():
    result = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'raw_files': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((root / 'data/raw').rglob('*')) if p.is_file()},
              'source_sha256': {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                                for p in sorted((root / 'src').rglob('*.py'))}}
    # 只读事务，保存每行摘要而不导出已有用户对话内容。
    for label, path, tables in [('index', 'data/index/chroma.sqlite3', ['embeddings', 'embedding_metadata']),
                                 ('sessions', 'data/sessions/memory.sqlite3', ['sessions', 'messages', 'summaries', 'rag_history'])]:
        with sqlite3.connect((root / path).resolve().as_uri() + '?mode=ro', uri=True) as db:
            db.execute('BEGIN')
            result[label] = {table: sorted(hashlib.sha256(json.dumps(row, ensure_ascii=False, default=str).encode()).hexdigest()
                                           for row in db.execute('SELECT * FROM ' + table)) for table in tables}
    return result

if mode in ('before', 'after', 'after-ui'):
    result = snapshot()
    save(mode + '.json', result)
    print(json.dumps({'commit': result['commit'], 'raw_files': len(result['raw_files']),
                      'index_rows': {k: len(v) for k, v in result['index'].items()},
                      'session_rows': {k: len(v) for k, v in result['sessions'].items()}}, ensure_ascii=False))
else:
    environment = dict(os.environ, PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       TOKENIZERS_PARALLELISM='false', RAG_OLLAMA_BASE_URL='http://127.0.0.1:11434')
    commands = [
        ('regression', [sys.executable, 'reports/模块完整性验证/verify_completeness_tests.py', '--scope', 'all', '--output', str(output / 'regression.json')]),
        ('basic', [sys.executable, 'reports/5_5_3 Bad Case分析与优化/verify_basic_tasks.py', '--base-url', 'http://127.0.0.1:11434', '--output', str(output / 'basic')])]
    run = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'started': datetime.now().isoformat(), 'exit_codes': {}}
    for label, command in commands:
        run['stage'] = label
        save('run.json', run)
        with (output / (label + '.log')).open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        run['exit_codes'][label] = completed.returncode
    run.update(stage='finished', finished=datetime.now().isoformat())
    save('run.json', run)
    print(json.dumps(run, ensure_ascii=False))
    raise SystemExit(int(any(run['exit_codes'].values())))
