"""Windows真实HTTP故障与线程生命周期核验；故障代理不提供伪造模型答案。"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sys
from threading import Event, Thread
from time import perf_counter, sleep
from unittest.mock import patch
from urllib.request import ProxyHandler, Request, build_opener


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.root.exists() or args.output.exists():
        raise FileExistsError('保留已有故障证据，请使用新目录')
    args.root.mkdir(parents=True)
    sys.path.insert(0, str(args.project))
    from langchain_core.messages import BaseMessage
    from langchain_core.tools import tool
    import src.agent.router as router
    from src.agent.react_loop import run_react
    from src.generation.rag_pipeline import build_context
    from src.generation.streaming import stream_answer
    from src.utils.config import load_config

    def serial(value):
        if isinstance(value, BaseMessage):
            return value.model_dump()
        if isinstance(value, dict):
            return {key: serial(item) for key, item in value.items()}
        if isinstance(value, list):
            return [serial(item) for item in value]
        return value

    config = deepcopy(load_config())
    config['paths']['logs'] = str(args.root / 'logs')
    real_url = config['llm']['base_url'].rstrip('/')
    opener = build_opener(ProxyHandler({})).open
    with opener(real_url + '/api/version', timeout=10) as response:
        version = json.load(response)
    report = {'started_at': datetime.now().astimezone().isoformat(), 'scope': __doc__,
              'ollama': version, 'rows': [], 'http_requests': [], 'passed': False}
    released, entered = Event(), Event()
    mode = {'value': 'normal'}

    def save():
        args.output.write_text(json.dumps(serial(report), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    class Relay(BaseHTTPRequestHandler):
        """转发真实服务；中断或延迟只在测试专用回环端口生效。"""
        def log_message(self, *_):
            pass

        def do_GET(self):
            entered.set()
            report['http_requests'].append({'method': 'GET', 'path': self.path, 'mode': mode['value']})
            if self.path == '/failure':
                self.send_error(503, 'Explicit acceptance fault')
                return
            if mode['value'] == 'delay':
                released.wait(timeout=20)
            with opener(real_url + self.path, timeout=10) as upstream:
                data = upstream.read()
            try:
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:
                pass  # 客户端确实已经因HTTP超时关闭，后续结果不能进入原请求。

        def do_POST(self):
            data = self.rfile.read(int(self.headers['Content-Length']))
            report['http_requests'].append({'method': 'POST', 'path': self.path, 'mode': mode['value'],
                                           'request': json.loads(data)})
            request = Request(real_url + self.path, data=data, headers={'Content-Type': 'application/json'})
            with opener(request, timeout=300) as upstream:
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson')
                self.send_header('Connection', 'close')
                self.end_headers()
                if mode['value'] == 'disconnect':
                    # 只转发一个含真实正文的包，随后关闭；从未制造done完成包。
                    for line in upstream:
                        self.wfile.write(line)
                        self.wfile.flush()
                        if json.loads(line).get('message', {}).get('content'):
                            break
                else:
                    self.wfile.write(upstream.read())
                self.close_connection = True

    server = ThreadingHTTPServer(('127.0.0.1', 0), Relay)
    server.daemon_threads = True
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    proxy = f'http://127.0.0.1:{server.server_port}'
    # 绑定后关闭形成真实的本机拒绝连接，不替换urllib调用。
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        unavailable = f'http://127.0.0.1:{reserve.getsockname()[1]}'
    config['llm']['base_url'] = proxy
    try:
        with ExitStack() as stack:
            for module in ('src.agent.router', 'src.agent.react_loop', 'src.generation.rag_pipeline',
                           'src.utils.logger'):
                stack.enter_context(patch(module + '.load_config', return_value=config))
            # 模型服务不可达时Agent必须明确失败，恢复地址后同样问题真实推理成功。
            config['llm']['base_url'] = unavailable
            events = list(run_react('计算9乘以8', stream=True))
            row = {'name': 'service_unavailable', 'events': events}
            row['passed'] = (events[-1]['task_complete'] is False and
                             any(e['type'] == 'error' for e in events) and
                             not any(e['type'] == 'tool_result' for e in events))
            report['rows'].append(row); save()
            assert row['passed']
            config['llm']['base_url'] = real_url
            events = list(run_react('计算9乘以8', stream=True))
            row = {'name': 'service_recovered', 'events': events,
                   'passed': events[-1]['task_complete'] and '72' in events[-1]['full_response']}
            report['rows'].append(row); save()
            assert row['passed']

            config['llm']['base_url'] = proxy
            mode['value'] = 'disconnect'
            events = list(stream_answer('用一句话说明Transformer。', build_context('用一句话说明Transformer。', [])))
            row = {'name': 'real_stream_disconnect', 'events': events,
                   'passed': bool(events[-1].get('raw_answer')) and events[-1]['type'] == 'error'
                             and not any(e['type'] == 'done' for e in events)}
            report['rows'].append(row); save()
            assert row['passed']

            # 工作线程内真实HTTP比外部预算更久；连续请求不能重复创建失控线程。
            mode['value'] = 'delay'
            config['agent'].update(max_parallel_calls=1, tool_timeout_seconds=.3)
            @tool
            def slow_service() -> dict:
                """读取测试代理后的真实Ollama版本，代理主动延迟返回。"""
                with opener(proxy + '/api/version', timeout=10) as response:
                    return json.load(response)
            call = {'name': slow_service.name, 'args': {}, 'call_id': 'slow-1'}
            before = len(report['http_requests'])
            started = perf_counter()
            outcomes = [list(router.execute_calls([{**call, 'call_id': f'slow-{i}'}], [slow_service]))[0]
                        for i in range(3)]
            snapshot = deepcopy(outcomes)
            row = {'name': 'deadline_capacity_lifecycle', 'results': outcomes,
                   'seconds_before_release': perf_counter() - started,
                   'active_before_release': router._active_executions,
                   'actual_http_calls': len(report['http_requests']) - before}
            released.set()
            until = perf_counter() + 10
            while router._active_executions and perf_counter() < until:
                sleep(.02)
            row['active_after_release'] = router._active_executions
            mode['value'] = 'normal'
            # 恢复验证使用正常HTTP预算；0.3秒仅用于前面的主动截止故障。
            config['agent']['tool_timeout_seconds'] = 10
            recovered = list(router.execute_calls([{**call, 'call_id': 'recovered'}], [slow_service]))[0]
            row['recovered'] = recovered
            row['passed'] = ([o['error_kind'] for o in outcomes] == ['deadline', 'capacity', 'capacity']
                             and row['active_before_release'] == row['actual_http_calls'] == 1
                             and outcomes == snapshot and row['active_after_release'] == 0
                             and recovered['status'] == 'success' and recovered['result'] == version)
            report['rows'].append(row); save()
            assert row['passed']

            # 原顺序收取结果：一个真实连接故障不能抹掉同批另一个真实服务返回。
            config['agent'].update(max_parallel_calls=2, tool_timeout_seconds=10)
            @tool
            def failed_service() -> dict:
                """测试代理主动返回HTTP 503，明确注入主服务执行故障。"""
                with opener(proxy + '/failure', timeout=10) as response:
                    return json.load(response)
            outcomes = list(router.execute_calls([
                {'name': failed_service.name, 'args': {}, 'call_id': 'failed'},
                {**call, 'call_id': 'success'}], [failed_service, slow_service], parallel=True))
            row = {'name': 'parallel_partial_failure', 'results': outcomes,
                   'passed': [o['status'] for o in outcomes] == ['error', 'success']
                             and outcomes[1]['result'] == version
                             and all(o['execution_mode'] == 'parallel' for o in outcomes)
                             and len(outcomes[0]['attempts']) == 1}
            report['rows'].append(row); save()
            assert row['passed']
            # 实际模型重新规划同一任务的替代工具；恢复前后工具均发真实HTTP。
            config['llm']['base_url'] = real_url
            config['agent']['tool_timeout_seconds'] = 300
            events = list(run_react('请先用failed_service查询Ollama版本；主工具执行失败时用slow_service查询同一个版本。',
                                    [failed_service, slow_service]))
            results = [e for e in events if e['type'] == 'tool_result']
            row = {'name': 'real_model_alternative_recovery', 'events': events,
                   'passed': events[-1]['task_complete'] and len(results) == 2
                             and [e['status'] for e in results] == ['error', 'success']
                             and results[1]['result'] == version
                             and any(e['type'] == 'recovery' for e in events)}
            report['rows'].append(row); save()
            assert row['passed']
        report['passed'] = all(row['passed'] for row in report['rows'])
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        released.set()
        server.shutdown(); server.server_close()
        report['finished_at'] = datetime.now().astimezone().isoformat()
        save()
        print(json.dumps({'passed': report['passed'], 'rows': [
            {'name': r['name'], 'passed': r['passed']} for r in report['rows']]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
