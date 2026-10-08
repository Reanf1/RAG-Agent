"""Windows 宿主机执行本轮独立容器探针，逐条保留真实退出码与输出。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--phase', choices=['seed', 'verify'], required=True)
args = parser.parse_args()
output = args.run / ('container-' + args.phase + '.json')
if output.exists():
    raise FileExistsError('不覆盖容器复测记录')
commands = []
if args.phase == 'seed':
    commands += [['docker', 'exec', 'rag-win-audit-20261008-app-1', 'python', '-m', 'pip', 'check']]
    # 镜像内文件逐项比对 Git 指纹，不能只依赖宿主机工作区。
    code = "import json,hashlib,pathlib; e=json.load(open('/proof/源码指纹.json')); rows=[dict(path=r['path'],passed=hashlib.sha256(pathlib.Path('/app',r['path']).read_bytes().replace(b'\\r\\n',b'\\n')).hexdigest()==r['lf_sha256']) for r in e['files'] if r['path'].startswith('src/') and r['path'].endswith('.py')]; print(json.dumps(rows)); assert rows and all(r['passed'] for r in rows)"
    commands += [['docker', 'exec', 'rag-win-audit-20261008-app-1', 'python', '-c', code]]
    commands += [['docker', 'network', 'inspect', 'rag-win-audit-20261008_offline']]
commands += [['docker', 'exec', 'rag-win-audit-20261008-app-1', 'python', '/proof/verify_container_runtime.py', args.phase]]
if args.phase == 'seed':
    commands += [['docker', 'compose', '-p', 'rag-win-audit-20261008', '-f', str(args.run / 'docker-compose.windows-audit.yml'), 'restart']]
else:
    commands += [['docker', 'exec', 'rag-win-audit-20261008-ollama-1', 'bash', '-c', "timeout 3 bash -c 'echo >/dev/tcp/1.1.1.1/443'; status=$?; echo outbound_status=$status; test $status -ne 0"]]
    commands += [['docker', 'inspect', 'rag-win-audit-20261008-app-1', 'rag-win-audit-20261008-ollama-1', 'rag-win-audit-20261008-gateway-1']]
results = []
for command in commands:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    row = {'command': command, 'exit_code': result.returncode,
           'stdout': result.stdout.decode('utf-8', errors='replace'),
           'stderr': result.stderr.decode('utf-8', errors='replace')}
    results.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    if result.returncode:
        break
passed = len(results) == len(commands) and all(r['exit_code'] == 0 for r in results)
with output.open('x', encoding='utf-8') as stream:
    json.dump({'checked_at': datetime.now().astimezone().isoformat(), 'phase': args.phase,
               'results': results, 'passed': passed}, stream, ensure_ascii=False, indent=2)
raise SystemExit(0 if passed else 1)
