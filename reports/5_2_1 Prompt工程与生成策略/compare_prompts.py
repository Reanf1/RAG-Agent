"""同一论文片段、问题与采样种子，对比基础问答模板和当前科研模板。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from langchain_core.messages import HumanMessage
from src.generation import rag_pipeline
from src.generation.prompt_template import PROMPT_VERSION, build_rag_messages
from src.utils.config import load_config
from compare_generation import evaluate_answer, summarize, SEEDS, RULE_VERSION


def basic_messages(question, context=''):
    """基线只提供相同资料与问题，不添加科研角色、数值条件和引用提醒。"""
    return [HumanMessage(content=f'请依据下面的资料回答问题。\n\n资料：\n{context}\n\n问题：{question}')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('实验结果不能覆盖')
    sample = json.loads(args.sample.read_text(encoding='utf-8'))
    assert sample['prompt_version'] == PROMPT_VERSION
    for case in sample['cases']:
        messages = [{'role': 'user' if m.type == 'human' else m.type, 'content': m.content}
                    for m in build_rag_messages(case['question'], case['context']['context'])]
        assert hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == case['messages_sha256']
    report = {'status': 'running', 'started_at': datetime.now().astimezone().isoformat(),
              'scope': '冻结相关片段的模板对照，不运行检索', 'prompt_version': PROMPT_VERSION,
              'dataset_sha256': hashlib.sha256(args.sample.read_bytes()).hexdigest(),
              'llm_config': load_config()['llm'], 'rule_version': RULE_VERSION, 'seeds': SEEDS,
              'basic_template': '请依据下面的资料回答问题。\n\n资料：\n{context}\n\n问题：{question}',
              'rows': []}
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report['warmup'] = rag_pipeline.generate_answer(sample['cases'][0]['question'], sample['cases'][0]['context'], options={'seed': 17})
    jobs = [(name, seed, case) for seed in SEEDS for case in sample['cases'] for name in ('basic', 'research')]
    random.Random(20261010).shuffle(jobs)
    save()
    for i, (name, seed, case) in enumerate(jobs, 1):
        started = perf_counter()
        builder = basic_messages if name == 'basic' else build_rag_messages
        # 两组只替换消息模板，来源、模型和响应解析均使用真实业务模块。
        with patch.object(rag_pipeline, 'build_rag_messages', builder):
            result = rag_pipeline.generate_answer(case['question'], case['context'], options={'seed': seed})
        report['rows'].append({'order': i, 'group': name, 'seed': seed, 'case_id': case['id'],
                              'wall_seconds': perf_counter()-started, 'result': result,
                              'checks': evaluate_answer(case, result)})
        save()
        print(f'{i}/{len(jobs)} {name} {case["id"]}', flush=True)
    report['summary'] = {name: summarize([r for r in report['rows'] if r['group'] == name]) for name in ('basic', 'research')}
    report.update(status='completed', finished_at=datetime.now().astimezone().isoformat())
    save()


if __name__ == '__main__':
    main()
