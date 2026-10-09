"""按序执行本轮补丁验证；所有结果单独保存，不覆盖完整120条。"""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['TEMP']) / 'rag-final-patch-20261009-c3b0fda'
root.mkdir(exist_ok=False)
source = Path(os.environ['TEMP']) / 'rag-final-quality-20261009-5117f57-corpus1692/runtime'
report = Path('reports/5_5_3 Bad Case分析与优化/最终质量收尾_20261009')
env = dict(os.environ, PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           TOKENIZERS_PARALLELISM='false', RAG_OLLAMA_BASE_URL='http://127.0.0.1:11434')
state = {'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'started_at':datetime.now().astimezone().isoformat(), 'status':'running', 'stages':[]}
for stage, command in [
    ('regression',[sys.executable,'reports/模块完整性验证/verify_completeness_tests.py','--scope','all','--output',str(root/'regression.json')]),
    ('parallel-after',[sys.executable,str(report/'probe_parallel_retrieval.py'),str(source/'index'),str(root/'parallel-after.json')]),
    ('targeted-agent',[sys.executable,str(report/'evaluate_targeted_agent.py'),str(source),str(root/'targeted-agent.json')]),
]:
    state['current_stage'] = stage
    (root/'run.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    with (root/(stage+'.log')).open('xb') as log:
        code = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    state['stages'].append({'stage':stage,'exit_code':code})
    if code:
        state['status']='failed'
        break
else:
    state['status']='completed'
state['ended_at']=datetime.now().astimezone().isoformat()
(root/'run.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
raise SystemExit(0 if state['status']=='completed' else 1)
