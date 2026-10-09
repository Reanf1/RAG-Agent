"""在Windows真实Ollama上复测已归档证据，只更新Prompt，不把冻结输入当检索验收。"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.generation import rag_pipeline
from src.generation.prompt_template import build_rag_messages
from src.utils.config import load_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--base-url', default='http://127.0.0.1:11435')
parser.add_argument('--only', nargs='*', help='只复测列出的题号；每轮独立留档')
parser.add_argument('--baseline', action='store_true', help='只复现上一提交的数量回答，参数与本轮一致')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
settings = load_config()
settings['llm']['base_url'] = args.base_url
builder = build_rag_messages
if args.baseline:
    namespace = {}
    source = subprocess.check_output(['git', 'show', '4a9e6c3:src/generation/prompt_template.py'], cwd=ROOT, text=True)
    exec(compile(source, 'baseline_prompt_template.py', 'exec'), namespace)
    builder = namespace['build_rag_messages']

previous = ROOT / 'reports/5_4_4 端到端联调与测试/Windows精简部署与预算调优_20261009'
audit = ROOT / 'reports/5_5_3 Bad Case分析与优化/独立审计修复_20261008/已知原页生成诊断'
cases = [('quantity', previous / 'fact-unit-prompt-replay.json', None)]
if not args.baseline:
    for name, number in [('F02',1), ('F11',2), ('F14',3), ('C05',4), ('C06',5), ('C14',6), ('S05',7), ('S06',8)]:
        cases.append((name, audit / f'http_{number:03d}.json', None))
    formal = ROOT / 'reports/5_5_2 系统性能评估/Windows正式性能评测_20261007/agent.calls.jsonl'
    for line in formal.open(encoding='utf-8'):
        row = json.loads(line)
        if row['id'] in {'R001', 'R007', 'R009', 'R015'} and row['profile'] == 'default' and row['request']['messages'][-1]['content'].startswith('【检索上下文】'):
            cases.append((row['id'], formal, row))

original_open = rag_pipeline.urlopen
for name, source, entry in cases:
    if args.only and name not in args.only:
        continue
    output = args.output / (name + '.json')
    if output.exists():
        raise FileExistsError('不覆盖已有迭代证据')
    old = entry or json.loads(source.read_text(encoding='utf-8'))
    content = old['request']['messages'][-1]['content']
    evidence, question = content.removeprefix('【检索上下文】\n').split('\n\n【用户问题】\n', 1)
    question = question.split('\n\n【作答提醒】', 1)[0]
    markers = list(re.finditer(r'\[参考文档(\d+) - 来源: (.*?)；原始块位置: (.*?)\]\n', evidence))
    references = [dict(id=int(m[1]), source_file=m[2], location=m[3], metadata={}, truncated='[正文已截断]' in evidence[m.end():markers[i+1].start() if i+1<len(markers) else len(evidence)],
                       text=evidence[m.end():markers[i+1].start() if i+1<len(markers) else len(evidence)].strip().removesuffix('\n[正文已截断]').strip()) for i,m in enumerate(markers)]
    context = dict(context=evidence, references=references, generation_mode='grounded')
    recording = {}
    # 转交同一真实HTTP，不替换模型；完整请求和原始响应随结果保存。
    def recorded(request, **kwargs):
        recording['request'] = json.loads(request.data)
        response = original_open(request, **kwargs)
        class Response:
            def __enter__(self):
                response.__enter__()
                return self
            def read(self):
                data = response.read()
                recording['response'] = json.loads(data)
                return data
            def __exit__(self, *values):
                return response.__exit__(*values)
        return Response()
    started = perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(rag_pipeline, 'load_config', return_value=deepcopy(settings)))
        stack.enter_context(patch.object(rag_pipeline, 'build_rag_messages', side_effect=builder))
        stack.enter_context(patch.object(rag_pipeline, 'urlopen', side_effect=recorded))
        try:
            result = dict(result=rag_pipeline.generate_answer(question, context, options={'seed':42}))
        except Exception as error:
            result = dict(error=repr(error))
    value = dict(case=name, baseline=args.baseline, source=str(source.relative_to(ROOT)), question=question,
                 context_sha256=hashlib.sha256(evidence.encode()).hexdigest(), seconds=perf_counter()-started,
                 boundary='真实Windows GPU模型，冻结原检索或已知原页证据；非盲评，不替代120题或生产检索验收。', **recording, **result)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(name, round(value['seconds'],3), result.get('error') or result['result'].get('warnings'), flush=True)
