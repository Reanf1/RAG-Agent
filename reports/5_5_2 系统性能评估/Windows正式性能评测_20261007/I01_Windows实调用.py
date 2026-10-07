"""单次重放F002的修复后观察请求，不改写冻结实验或宣称全量答案已重测。"""

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw_output = args.output.with_suffix('.response.jsonl')
    if args.output.exists() or raw_output.exists():
        raise FileExistsError('实际请求及响应不得覆盖历史证据')
    sys.path.insert(0, str(args.project))
    from src.utils.config import load_config
    from src.utils.token_budget import request_tokens

    payload = json.loads(args.request.read_text(encoding='utf-8'))
    # 冻结轨迹使用非流式请求；只切换响应传输方式，不改写输入消息或工具。
    payload['stream'] = True
    # 与冻结Agent实验使用同一采样seed；此请求仅重放最终观察阶段。
    payload['options']['seed'] = 20261003
    estimated = request_tokens(payload)
    budget = payload['options']['num_ctx'] - payload['options']['num_predict']
    assert estimated <= budget
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    endpoint = load_config()['llm']['base_url'].rstrip('/') + '/api/chat'
    report = {'case': 'F002:no_rules:observation_replay', 'status': 'running',
              'started_at': datetime.now().astimezone().isoformat(),
              'source_sha256': sha256((args.project / 'src/agent/react_loop.py').read_bytes()).hexdigest(),
              'original_request_sha256': sha256(args.request.read_bytes()).hexdigest(),
              'sent_request_sha256': sha256(data).hexdigest(), 'model_seed': 20261003,
              'estimated_tokens': estimated, 'input_budget': budget,
              'boundary': '仅单次观察请求实调用；未重做规划和检索，不覆盖120条冻结质量基线。'}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    started = perf_counter()
    content, chunks, done = [], 0, None
    with raw_output.open('xb') as output:
        with urlopen(Request(endpoint, data=data, headers={'Content-Type': 'application/json'}), timeout=300) as response:
            report['http_status'] = response.status
            for line in response:
                output.write(line)
                output.flush()
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get('error'):
                    raise RuntimeError(event['error'])
                piece = event.get('message', {}).get('content', '')
                if piece:
                    content.append(piece)
                    chunks += 1
                if event.get('done'):
                    done = event
    assert done and content and all(type(done.get(k)) is int for k in ('prompt_eval_count', 'eval_count'))
    report.update(status='passed', seconds=perf_counter()-started, stream_content_events=chunks,
                  actual_input_tokens=done['prompt_eval_count'], actual_output_tokens=done['eval_count'],
                  answer=''.join(content), done_reason=done.get('done_reason'),
                  response_sha256=sha256(raw_output.read_bytes()).hexdigest(),
                  completed_at=datetime.now().astimezone().isoformat())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
