"""只调用临时Ollama端口，比较相同长输入与新路径完整证据；不更改项目或冻结评测。"""
from pathlib import Path
from urllib.request import Request, urlopen
from time import perf_counter
import hashlib
import json
import os
import subprocess

import time


def main():
    root = Path(os.environ['TEMP'])
    endpoint = 'http://127.0.0.1:11436'
    output = root / 'rag-context-bench-20261009.json'
    if output.exists():
        raise FileExistsError('不覆盖已存在的测量结果')
    def get(path):
        with urlopen(endpoint + path, timeout=15) as r:
            return json.load(r)
    def post(payload):
        with urlopen(Request(endpoint + '/api/chat', data=json.dumps(payload, ensure_ascii=False).encode(),
                             headers={'Content-Type':'application/json'}), timeout=300) as r:
            return json.load(r)
    report = {'version': get('/api/version'), 'tags': get('/api/tags'), 'results': [],
              'boundary': '串行孤立服务，未加载Embedding/Reranker，未重跑120条质量集或部署应用'}
    for size in (8192, 16384):
        warm = post({'model':'qwen2.5:7b','stream':False,'messages':[{'role':'user','content':'回复OK'}],
                     'options':{'num_ctx':size,'num_predict':16,'seed':20261003,'temperature':0}})
        print(json.dumps({'warmup': size, 'done_reason':warm.get('done_reason')}, ensure_ascii=False), flush=True)
        for sample in (1, 2):
            raw = (root / f'rag-context-{size}.json').read_bytes()
            payload = json.loads(raw)
            started = perf_counter()
            response = post(payload)
            seconds = perf_counter()-started
            (root / f'rag-context-{size}-{sample}.response.json').write_text(json.dumps(response,ensure_ascii=False),encoding='utf-8')
            result = {'num_ctx':size,'sample':sample,'input_file_sha256':hashlib.sha256(raw).hexdigest(),
                      'seconds':seconds,'response':response, 'loaded':get('/api/ps'),
                      'gpu':subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()}
            report['results'].append(result)
            output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in result.items() if k not in ('response','loaded') } | {
                k:response.get(k) for k in ('done','done_reason','prompt_eval_count','eval_count','load_duration')},ensure_ascii=False),flush=True)
    raw = (root / 'rag-context-current-16384.json').read_bytes()
    started = perf_counter();response=post(json.loads(raw))
    report['current_full_evidence']={'seconds':perf_counter()-started,'input_file_sha256':hashlib.sha256(raw).hexdigest(),
        'response':response,'loaded':get('/api/ps')}
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'current_full_evidence_seconds':report['current_full_evidence']['seconds'],
                     'actual_input_tokens':response.get('prompt_eval_count'),'actual_output_tokens':response.get('eval_count'),
                     'output':str(output)},ensure_ascii=False),flush=True)


# SSH会话内持有临时服务，结束时卸载本轮模型并关闭自己创建的进程。
root = Path(os.environ['TEMP'])
log = (root / 'rag-context-20261009-serve.log').open('ab')
env = dict(os.environ, OLLAMA_HOST='127.0.0.1:11436')
server = subprocess.Popen([str(Path(os.environ['LOCALAPPDATA']) / 'Programs/Ollama/ollama.exe'), 'serve'], env=env, stdout=log, stderr=log)
print(json.dumps({'owned_server_pid':server.pid}),flush=True)
try:
    for attempt in range(20):
        if server.poll() is not None:
            raise RuntimeError('临时Ollama启动失败：'+(root / 'rag-context-20261009-serve.log').read_text(errors='replace')[-2000:])
        try:
            with urlopen('http://127.0.0.1:11436/api/version',timeout=2) as response:
                json.load(response)
            break
        except Exception:
            time.sleep(.5)
    main()
finally:
    try:
        payload = {'model':'qwen2.5:7b','keep_alive':0}
        with urlopen(Request('http://127.0.0.1:11436/api/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=15) as response:
            response.read()
    except Exception:
        pass
    server.terminate()
    server.wait(timeout=15)
    log.close()
