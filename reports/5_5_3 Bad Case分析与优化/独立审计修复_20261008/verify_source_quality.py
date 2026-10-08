"""按已知相关物理页隔离复测生成质量；不作为整链路召回成绩。"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from verify_real_repairs import save, observe_http
from src.utils import config

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
audit = Path('/Users/rean/github/测试项')
settings = config.load_config()
settings['llm']['base_url'] = 'http://127.0.0.1:11439'
# 诊断明确区分证据覆盖与生成错误：尽量容纳已知相关页，Token上限仍沿用8192。
settings['generation'].update(max_context_chars=18000, max_prompt_chars=22000)
settings['paths']['logs'] = str(args.output / 'business_logs')
config.load_config = lambda: deepcopy(settings)
from src.data_loader.pdf_loader import load_pdf
from src.generation.rag_pipeline import build_context, generate_answer

save(args.output / '运行配置.json', settings)
observe_http(args.output)
questions = json.loads((audit / 'docs/完整性测试证据/新增审计评测集60题.json').read_text())['questions']
papers = {}
for item in questions:
    if item['id'].split('-')[-1] not in {'F02', 'F11', 'F14', 'C05', 'C06', 'C14', 'S05', 'S06'}:
        continue
    selected, fingerprints, seen = [], {}, set()
    for evidence in item['evidence']:
        path, page = audit / evidence['pdf_path'], evidence['page_number']
        if (str(path), page) in seen:
            continue
        seen.add((str(path), page))
        if path not in papers:
            papers[path] = load_pdf(path)
        fingerprints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        # 输入为加载器实际正文；评测集只用于指定物理页，不把参考答案或quote输入模型。
        selected.extend((doc, 1.0) for doc in papers[path]
                        if doc.metadata['page_number'] == page and doc.metadata.get('content_type') != 'table')
    started = perf_counter()
    context = build_context(item['question'], selected)
    context['generation_mode'] = 'grounded'
    try:
        result = {'result': generate_answer(item['question'], context)}
    except Exception as error:
        result = {'error': repr(error)}
    save(args.output / (item['id'] + '.json'), {
        'question': item['question'], 'seconds': perf_counter() - started, 'context': context,
        'source_sha256': fingerprints,
        'boundary': '按已知相关页提供真实正文的生成诊断；非盲评、非召回或Agent整链路成绩。字符预算扩大以定位遗漏，Token预算不变。', **result})
    print(item['id'], '完成' if 'result' in result else result, flush=True)
