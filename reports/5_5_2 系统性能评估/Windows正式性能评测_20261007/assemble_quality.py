"""汇总逐条阅读后的显式评分，核验答案指纹；不使用算法打质量分。"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--judgments', type=Path, required=True)
parser.add_argument('--additional-pages', type=Path, required=True)
args = parser.parse_args()
agent = json.loads((HERE / 'agent.json').read_text(encoding='utf-8'))
notes = json.loads(args.judgments.read_text(encoding='utf-8'))
questions = {q['id']: q for q in json.loads((ROOT / 'reports/评测集.json').read_text(encoding='utf-8'))}
keys = {(q, p) for q in questions for p in ('default', 'no_rules')}
assert agent['status'] == 'completed' and len(agent['rows']) == 120
assert {(r['id'], r['profile']) for r in agent['rows']} == keys
assert set(notes) == {q + ':' + p for q, p in keys}
rows = []
for answer in agent['rows']:
    q, note = questions[answer['id']], notes[answer['id'] + ':' + answer['profile']]
    digest = sha256(answer['answer'].encode('utf-8')).hexdigest()
    assert digest == note['answer_sha256']
    assert len(note['scores']) == 3 and all(type(s) is int and 0 <= s <= 4 for s in note['scores'])
    assert note['reason'].strip()
    rows.append({'id': answer['id'], 'profile': answer['profile'], 'category': q['category'],
                 'question': q['question'], 'answer': answer['answer'], 'answer_sha256': digest,
                 'reference_answer': q['reference_answer'], 'answer_points': q['answer_points'],
                 'evidence': q['evidence'],
                 'assistant_scores': dict(zip(('correctness', 'completeness', 'citation_accuracy'), note['scores'])),
                 'assistant_reason': note['reason'],
                 'user_review': dict.fromkeys(('correctness', 'completeness', 'citation_accuracy', 'reason', 'reviewer', 'date'))})
summary = {}
for profile in ('default', 'no_rules'):
    selected = [r for r in rows if r['profile'] == profile]
    assert len(selected) == 60
    summary[profile] = {'count': 60, 'user_reviewed': 0,
                        'means': {k: statistics.mean(r['assistant_scores'][k] for r in selected)
                                  for k in selected[0]['assistant_scores']},
                        'core_unanswered_or_wrong': sum(r['assistant_scores']['correctness'] == 0 for r in selected),
                        'without_valid_citation': sum(r['assistant_scores']['citation_accuracy'] == 0 for r in selected)}
old_pages = ROOT / 'reports/5_5_2 系统性能评估/助手初评原文证据_20261006.json'
pages = json.loads(old_pages.read_text(encoding='utf-8'))['pages'] + json.loads(args.additional_pages.read_text(encoding='utf-8'))
for page in pages:
    assert sha256((ROOT / page['source_path']).read_bytes()).hexdigest() == page['pdf_sha256']
sources = [HERE / 'agent.json', ROOT / 'reports/评测集.json', old_pages,
           args.judgments, args.additional_pages]
result = {'reviewer': 'Codex助手逐题初评，用户终审待进行', 'date': '2026-10-07',
          'scope': '本轮Windows冻结源码120条实际最终答案；修复前失败保留，不是修复后全量重跑或独立盲评',
          'method': ['逐条阅读原答案、评分要点和对应原论文物理页后显式给分。',
                     '脚本只核验指纹与汇总，不自动判断质量；工具中间结果不代替最终答案。',
                     '助手参与了评测集构建，初评不是独立盲评；用户审核栏全部空白。'],
          'source_fingerprints': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:
                                  sha256(p.read_bytes()).hexdigest() for p in sources},
          'pdf_evidence_pages': len({(p['paper_id'], p['physical_page']) for p in pages}),
          'summary': summary, 'rows': rows}
output = HERE / '助手逐题初评.json'
if output.exists():
    raise FileExistsError('不覆盖已归档初评')
assert Counter(r['category'] for r in rows) == dict.fromkeys(('fact', 'comparison', 'synthesis', 'reasoning'), 30)
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'records': len(rows), 'summary': summary}, ensure_ascii=False, indent=2))
