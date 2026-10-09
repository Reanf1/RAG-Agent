"""核对最终镜像源码与会话计时；模型HTTP隔离，SQLite实际落盘。"""
from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, '/app')
expected = json.loads(Path('/proof/source-history.json').read_text(encoding='utf-8'))
checks = []
for row in expected['files']:
    # Dockerfile和Compose只用于构建／编排；镜像仅含运行源码、配置及requirements。
    if row['path'].startswith('tests/') or row['path'] in {'docker/Dockerfile', 'docker/docker-compose.yml'}:
        continue
    actual = Path('/app') / row['path']
    checks.append({'path': row['path'], 'passed': sha256(actual.read_bytes()).hexdigest() == row['lf_sha256']})
assert all(item['passed'] for item in checks), '最终镜像源码不一致'
suite = unittest.defaultTestLoader.loadTestsFromName('tests.5_3_4 多轮对话记忆管理.test_session_isolation')
result = unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps({'source_revision': expected['source_revision'], 'checks': checks,
                  'tests_run': result.testsRun, 'failures': len(result.failures),
                  'errors': len(result.errors), 'skipped': len(result.skipped),
                  'passed': result.wasSuccessful(),
                  'boundary': '真实镜像与SQLite会话核验，模型HTTP隔离；此前独立容器真实模型证据不重写'}, ensure_ascii=False))
raise SystemExit(0 if result.wasSuccessful() else 1)
