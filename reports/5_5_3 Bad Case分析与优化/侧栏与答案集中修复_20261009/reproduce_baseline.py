"""在临时目录复现旧提交缺陷；叠加本轮回归用例，不交换工作区源码或数据。"""
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[3]
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True, help='新建证据目录，不覆盖归档结果')
OUTPUT = parser.parse_args().output
OUTPUT.mkdir(parents=True, exist_ok=True)
names = ['test_two_paper_rule_rejects_missing_or_wrong_target_before_execution',
         'test_one_paper_result_cannot_complete_two_paper_request',
         'test_page_startup_shows_sidebar_status_and_rerun_reuses_snapshot']
with tempfile.TemporaryDirectory(prefix='rag-status-baseline-') as directory:
    temporary = Path(directory)
    archive = subprocess.check_output(['git','archive','4a9e6c3','src','tests','config.yaml'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as package:
        package.extractall(temporary)
    # 回归需现有只读分词器；不连接生产索引、原文、会话或日志目录。
    (temporary / 'data/models').mkdir(parents=True)
    (temporary / 'data/models/qwen2.5-tokenizer').symlink_to(ROOT / 'data/models/qwen2.5-tokenizer', target_is_directory=True)
    for file in ('5_3_1 Agent核心循环/test_observation.py',
                 '5_3_3 Agent决策优化/test_routing_parallel.py','5_4_2 可观测性与健康检查/test_health_check.py'):
        shutil.copy2(ROOT / 'tests' / file, temporary / 'tests' / file)
    code = '''import json, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
def flatten(suite):
 for item in suite:
  if isinstance(item, unittest.TestSuite): yield from flatten(item)
  else: yield item
names = json.loads(sys.argv[1])
tests = [t for t in flatten(unittest.TestLoader().discover('tests')) if t._testMethodName in names]
r = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(tests))
Path(sys.argv[2]).write_text(json.dumps(dict(revision='4a9e6c3', run=r.testsRun, tests=[t.id() for t in tests], failures=[(t.id(),e) for t,e in r.failures], errors=[(t.id(),e) for t,e in r.errors], boundary='旧提交真实源码与本轮缺陷用例，模型响应隔离；预期复现失败，不作为成功回归。'),ensure_ascii=False,indent=2)+'\\n',encoding='utf-8')
'''
    with (OUTPUT / 'baseline-defects-confirmed.log').open('x',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-c',code,json.dumps(names),str(OUTPUT/'baseline-defects-confirmed.json')],cwd=temporary,stdout=log,stderr=subprocess.STDOUT,check=True)
print('旧提交复现已保存')
