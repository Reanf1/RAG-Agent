"""在隔离副本依次运行Windows正式实验，保存资源采样和每阶段退出状态。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    import psutil
    stop = Event()
    report = {'started_at': datetime.now().astimezone().isoformat(), 'status': 'running',
              'project': str(args.project), 'python': sys.executable, 'stages': []}

    def save():
        (args.root / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    def sample():
        # 五秒一次，采样不读取用户文档或其他进程命令行；模型进程单独统计。
        psutil.cpu_percent()
        with (args.root / 'resources.jsonl').open('x', encoding='utf-8') as output:
            while not stop.wait(5):
                processes = []
                for process in psutil.process_iter(['pid', 'name', 'memory_info']):
                    try:
                        if process.info['name'].lower() in ('python.exe', 'ollama.exe', 'ollama_llama_server.exe'):
                            processes.append({'pid': process.pid, 'name': process.info['name'],
                                              'rss_bytes': process.info['memory_info'].rss})
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                row = {'time': datetime.now().astimezone().isoformat(), 'cpu_percent': psutil.cpu_percent(),
                       'memory': psutil.virtual_memory()._asdict(), 'processes': processes}
                try:
                    gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                    row['gpu'] = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
                except (OSError, subprocess.TimeoutExpired) as error:
                    row['gpu_error'] = repr(error)
                output.write(json.dumps(row, ensure_ascii=False)+'\n'); output.flush()

    worker = Thread(target=sample, daemon=True)
    worker.start(); save()
    directory = args.project / 'reports' / '5_5_2 系统性能评估'
    frozen = args.root / 'system-runtime'
    stages = [('retrieval', 'evaluate_system.py', frozen), ('agent', 'evaluate_system.py', frozen),
              ('routing', 'evaluate_research.py', args.root / 'routing-runtime'),
              ('parallel', 'evaluate_research.py', args.root / 'parallel-runtime')]
    try:
        for stage, script, runtime in stages:
            command = [sys.executable, str(directory / script), '--stage', stage,
                       '--root', str(runtime), '--output', str(args.root / (stage+'.json'))]
            if script == 'evaluate_research.py':
                command.extend(['--frozen-root', str(frozen)])
            row = {'stage': stage, 'started_at': datetime.now().astimezone().isoformat(), 'command': command}
            report['stages'].append(row); save()
            print(f'开始正式阶段：{stage}', flush=True)
            started = perf_counter()
            with (args.root / (stage+'.log')).open('xb') as output:
                result = subprocess.run(command, cwd=args.project, stdout=output, stderr=subprocess.STDOUT)
            row.update(exit_code=result.returncode, seconds=perf_counter()-started,
                       finished_at=datetime.now().astimezone().isoformat()); save()
            if result.returncode:
                raise RuntimeError(f'{stage}退出码{result.returncode}，保留本轮原始结果')
            print(f'完成正式阶段：{stage}，{row["seconds"]:.2f}秒', flush=True)
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        stop.set(); worker.join(timeout=5)
        report['finished_at'] = datetime.now().astimezone().isoformat(); save()


if __name__ == '__main__':
    main()
