"""独立复算新增实验：从逐题排名及实际Ollama响应核验，不信任汇总数。"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['chunking', 'routing', 'parallel'])
    args = parser.parse_args()
    paths = {'chunking': ROOT/'reports/5_1_2 文本分块策略/五组分块检索对比_20261004.json',
             'routing': ROOT/'reports/5_5_2 系统性能评估/Agent与固定RAG对照_20261004.json',
             'parallel': ROOT/'reports/5_5_2 系统性能评估/独立工具串行并行对照_20261004.json'}
    path = paths[args.stage]
    data = json.loads(path.read_text())
    assert data['status'] == 'completed', '实验尚未完成，不能作为最终结论'
    assert len(data['rows']) == {'chunking': 300, 'routing': 144, 'parallel': 12}[args.stage]
    errors, checks = [], 0
    if args.stage == 'chunking':
        questions = {q['id']: q for q in json.loads((ROOT/'reports/评测集.json').read_text())}
        papers = json.loads((ROOT/'reports/5_5_1 评测集构建/论文清单.json').read_text())['papers']
        ids = {p['id']: p['doc_id'] for p in papers}
        assert hashlib.sha256((ROOT/'reports/评测集.json').read_bytes()).hexdigest() == data['dataset_sha256']
        assert len({(r['id'], r['profile']) for r in data['rows']}) == 300
        for row in data['rows']:
            gold = {(ids[e['paper_id']], e['page_number']) for e in questions[row['id']]['evidence']}
            for k in (3, 5, 10):
                found, first = set(), None
                for rank, result in enumerate(row['top20'][:k], 1):
                    meta = result['metadata']
                    matched = {(meta['doc_id'], p) for p in range(meta['page_number'], meta.get('page_end', meta['page_number']) + 1)} & gold
                    found.update(matched)
                    if matched and first is None: first = rank
                expected = {'hit': bool(found), 'mrr': 1/first if first else 0, 'recall': len(found)/len(gold)}
                assert all(abs(expected[key] - row['at_k'][str(k)][key]) < 1e-12 for key in expected)
                checks += 1
        for name, profile in data['profiles'].items():
            rows = [r for r in data['rows'] if r['profile'] == name]
            assert len(rows) == profile['questions'] == 60
            for k in (3, 5, 10):
                for key in ('hit', 'mrr', 'recall'):
                    assert abs(mean(r['at_k'][str(k)][key] for r in rows) - profile['at_k'][str(k)][key]) < 1e-12
    else:
        raw = [json.loads(line) for line in path.with_suffix('.calls.jsonl').read_text().splitlines()]
        keys = [(r['id'], r['profile'], r['repeat']) for r in data['rows']]
        assert len(set(keys)) == len(keys)
        assert {tuple(c[k] for k in ('id', 'profile', 'repeat')) for c in raw} <= set(keys)
        for row in data['rows']:
            calls = [c for c in raw if tuple(c[k] for k in ('id', 'profile', 'repeat')) == (row['id'], row['profile'], row['repeat'])]
            missing = sum(any(type(c.get('response', {}).get(k)) is not int for k in ('prompt_eval_count', 'eval_count')) for c in calls)
            actual = {k: sum(c.get('response', {}).get(k, 0) or 0 for c in calls) for k in ('prompt_eval_count', 'eval_count')}
            assert row['tokens']['unknown_calls'] == missing
            assert all(row['tokens'][k] == v for k, v in actual.items())
            assert row['tokens']['total'] == (None if missing else sum(actual.values()))
            checks += len(calls)
            if args.stage == 'routing' and row['profile'] == 'fixed_rag':
                assert [c['name'] for c in row['tool_calls']] == ['knowledge_base_search']
            if args.stage == 'parallel':
                # 规划和执行必须确有两项独立调用；失败也保留，不能冒充并行完成。
                first = [c for c in row['tool_calls'] if c['iteration'] == 1]
                results = [c for c in row['tool_results'] if c['iteration'] == 1]
                valid = len(first) == len(results) == 2 and len({json.dumps((c['name'], c['args']), sort_keys=True) for c in first}) == 2
                valid = valid and all(c['status'] == 'success' and c['execution_mode'] == row['profile'] for c in results)
                if not valid: errors.append({'id': row['id'], 'profile': row['profile'], 'repeat': row['repeat'], 'reason': '未形成两项独立且成功的目标调度调用'})
        for name, profile in data['profiles'].items():
            rows = [r for r in data['rows'] if r['profile'] == name]
            known = [r['tokens']['total'] for r in rows if r['tokens']['total'] is not None]
            assert profile['requests'] == len(rows)
            assert abs(mean(r['seconds'] for r in rows) - profile['seconds_mean']) < 1e-9
            assert profile['tokens_total_known'] == sum(known)
    output = {'stage': args.stage, 'status': 'passed' if not errors else 'completed_with_schedule_failures',
              'rows': len(data['rows']), 'independent_checks': checks, 'schedule_failures': errors,
              'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    target = path.with_name(path.stem + '_独立复核.json')
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__': main()
