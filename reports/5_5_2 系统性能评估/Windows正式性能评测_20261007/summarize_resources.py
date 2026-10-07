"""按已完成阶段汇总五秒资源采样；整机负载不冒充单请求资源开销。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
from statistics import mean


def summarize(rows):
    """没有GPU读数时保留未知，不以0填补；内存统一使用GiB/MiB。"""
    gpu = []
    for row in rows:
        fields = row.get('gpu', '').split(',')
        if len(fields) == 3:
            try:
                gpu.append([float(value.strip()) for value in fields])
            except ValueError:
                pass
    peak_by_pid = {}
    for row in rows:
        for process in row['processes']:
            key = str(process['pid'])
            old = peak_by_pid.get(key, {}).get('rss_peak_gib', 0)
            peak_by_pid[key] = {'name': process['name'],
                                'rss_peak_gib': max(old, process['rss_bytes'] / 2**30)}
    return {'samples': len(rows),
            'cpu_percent_mean': mean(r['cpu_percent'] for r in rows),
            'cpu_percent_peak_sampled': max(r['cpu_percent'] for r in rows),
            'ram_used_gib_mean': mean(r['memory']['used'] / 2**30 for r in rows),
            'ram_used_gib_peak_sampled': max(r['memory']['used'] / 2**30 for r in rows),
            'ram_percent_mean': mean(r['memory']['percent'] for r in rows),
            'gpu_valid_samples': len(gpu),
            'gpu_utilization_percent_mean': mean(g[0] for g in gpu) if gpu else None,
            'gpu_memory_mib_peak_sampled': max(g[1] for g in gpu) if gpu else None,
            'process_rss_peak_by_pid': peak_by_pid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    target = args.root / '资源采样汇总.json'
    if target.exists():
        raise FileExistsError('不覆盖已有资源统计')
    run = json.loads((args.root / 'run.json').read_text(encoding='utf-8'))
    assert run['status'] == 'completed' and len(run['stages']) == 4
    rows = [json.loads(line) for line in (args.root / 'resources.jsonl').read_text(encoding='utf-8').splitlines()]
    assert rows and all(stage['exit_code'] == 0 for stage in run['stages'])
    times = [datetime.fromisoformat(row['time']) for row in rows]
    assert times == sorted(times)
    result = {'status': 'completed', 'overall': summarize(rows), 'stages': {},
              'boundary': '约五秒整机采样，峰值仅为采样点最大值；原生页面等既有服务保持运行，不能归因于单个请求。阶段含预热；检索阶段还含建库。GPU内存为MiB，RAM与RSS为GiB。'}
    for stage in run['stages']:
        start, finish = [datetime.fromisoformat(stage[key]) for key in ('started_at', 'finished_at')]
        selected = [row for row, moment in zip(rows, times) if start <= moment <= finish]
        assert selected, stage['stage']
        result['stages'][stage['stage']] = summarize(selected)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value['samples'] for key, value in result['stages'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
