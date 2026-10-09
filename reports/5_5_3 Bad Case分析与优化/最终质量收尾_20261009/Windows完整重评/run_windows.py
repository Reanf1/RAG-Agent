"""冻结源码后一次运行Windows回归、隔离建库和120条完整Agent重评。"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--passed-regression', type=Path, help='同一冻结提交已经通过的回归报告，避免重复运行')
args = parser.parse_args()
root = args.root
root.mkdir(parents=True, exist_ok=False)
project = Path.cwd()
env = dict(os.environ, PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           TOKENIZERS_PARALLELISM='false', RAG_OLLAMA_BASE_URL='http://127.0.0.1:11434')
state = {'started_at':datetime.now().astimezone().isoformat(), 'status':'running',
         'commit':subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
         'boundary':'索引重新构建；5组各1题只作准备核验，不作为检索性能。60题两组Agent共120条重新检索和生成，失败保留。',
         'stages':[]}
if args.passed_regression:
    previous = json.loads(args.passed_regression.read_text(encoding='utf-8'))
    assert previous['passed'] and previous['run'] == 847
    previous_commit = json.loads((args.passed_regression.parent/'run.json').read_text(encoding='utf-8'))['commit']
    # 只允许评测脚本修正后复用业务回归；业务源码、配置和测试必须完全相同。
    assert not subprocess.check_output(['git','diff',previous_commit,'HEAD','--','src','tests','config.yaml'])
    state['reused_regression'] = str(args.passed_regression)
    (root/'regression.json').write_bytes(args.passed_regression.read_bytes())
for label, command in [
    ('regression',[sys.executable,'reports/模块完整性验证/verify_completeness_tests.py','--scope','all','--output',str(root/'regression.json')]),
    ('prepare',[sys.executable,'reports/5_5_2 系统性能评估/evaluate_system.py','--stage','retrieval','--root',str(root/'runtime'),'--output',str(root/'prepare.json'),'--limit','1']),
    ('agent',[sys.executable,'reports/5_5_2 系统性能评估/evaluate_system.py','--stage','agent','--root',str(root/'runtime'),'--output',str(root/'agent.json'),'--limit','60']),
]:
    if label == 'regression' and args.passed_regression:
        state['stages'].append({'stage':label, 'exit_code':0, 'reused_same_commit':True})
        continue
    state['current_stage'] = label
    (root/'run.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    with (root/(label+'.log')).open('xb') as log:
        code = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    state['stages'].append({'stage':label,'exit_code':code})
    if code:
        state['status']='failed'
        break
else:
    state['status']='completed'
state['ended_at']=datetime.now().astimezone().isoformat()
(root/'run.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
raise SystemExit(0 if state['status']=='completed' else 1)
