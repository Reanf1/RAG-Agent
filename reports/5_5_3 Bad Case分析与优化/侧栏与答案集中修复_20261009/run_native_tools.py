"""Windows真实两篇论文元信息调用；副本路径隔离，不更改原知识库。"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys
import subprocess
from time import perf_counter
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project', type=Path, required=True)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--name', default='two-paper')
args = parser.parse_args()
sys.path.insert(0, str(args.project))
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
from src.utils.config import load_config
from src.agent.tools import get_available_tools
from src.agent.react_loop import run_react
from langchain_core.messages import message_to_dict
modules = [importlib.import_module('src.' + name) for name in (
    'agent.react_loop','agent.tools','agent.router','generation.rag_pipeline',
    'retrieval.vector_store','utils.logger','utils.token_budget')]
config = deepcopy(load_config())
for key, relative in [('raw_documents','probe/raw'),('vector_index','probe/index'),('logs','two-paper-logs')]:
    config['paths'][key] = str(args.run / relative)
targets = ['8ce7b83971a14508ca711a27c875c9b6914c4f6767cf3150fb1ca6c07aa056d6',
           'bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697']
question = '分别读取论文ID ' + ' 和 '.join(targets) + ' 的元信息，分别给出标题与来源。'
started = perf_counter()
with ExitStack() as stack:
    for module in modules:
        if hasattr(module, 'load_config'):
            stack.enter_context(patch.object(module,'load_config',return_value=config))
    events = [event for event in run_react(question, get_available_tools(), stream=True) if event['type'] != 'token']
results = [event for event in events if event['type']=='tool_result' and event.get('name')=='paper_metadata' and event.get('status')=='success']
covered = {event.get('args',{}).get('doc_id') for event in results}
passed = len(results)==2 and covered==set(targets) and events[-1].get('task_complete') is True
value = dict(question=question,targets=targets,seconds=perf_counter()-started,events=events,passed=passed,
             revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.project,text=True).strip(),
             boundary='真实Ollama与原文元信息、隔离索引副本；只核对两次执行和目标覆盖，不代表论文问答质量全通过。')
with (args.run/(args.name+'.json')).open('x',encoding='utf-8') as output:
    json.dump(value,output,ensure_ascii=False,indent=2,default=message_to_dict)
print(json.dumps(dict(passed=passed,results=len(results),covered=sorted(covered),seconds=value['seconds']),ensure_ascii=False))
raise SystemExit(0 if passed else 1)
