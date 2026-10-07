"""重放已归档F002的请求组装，只检验预算和原始证据，不调用模型。"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from langchain_core.messages import AIMessage, ToolMessage
from src.agent import react_loop
from src.agent.tools import get_available_tools
from src.utils.config import generation_options, load_config
from src.utils.messages import messages_to_ollama
from src.utils.token_budget import request_tokens

row = next(r for r in json.loads((HERE / 'agent.json').read_text(encoding='utf-8'))['rows']
           if r['id'] == 'F002' and r['profile'] == 'no_rules')
calls = [json.loads(line) for line in (HERE / 'agent.calls.jsonl').read_text(encoding='utf-8').splitlines()]
call = next(c for c in calls if c['id'] == 'F002' and c['profile'] == 'no_rules'
            and [t['function']['name'] for t in c['request'].get('tools', [])] == ['paper_summary'])
state = json.loads(next(m['content'] for m in call['request']['messages'] if m['role'] == 'user'))
event = {k: v for k, v in [e for e in row['metrics']['trace'] if e['type'] == 'tool_result'][-1].items()
         if k not in {'type', 'iteration'}}
context = state['context']
context['observations'].append(event)
messages = [AIMessage(content='', tool_calls=[{'name': event['name'], 'args': event['args'], 'id': event['call_id']}]),
            ToolMessage(content=json.dumps(event['result'], ensure_ascii=False),
                        tool_call_id=event['call_id'], name=event['name'])]
question = next(q['question'] for q in json.loads((ROOT / 'reports/评测集.json').read_text(encoding='utf-8'))
                if q['id'] == 'F002')
model_request = react_loop._model_request

class Captured(Exception):
    """保存请求后结束只读回放，避免进入真实模型网络调用。"""

def capture(model_messages, **fields):
    config = load_config()['llm']
    before = {'model': config['model'], 'stream': False, 'options': generation_options(config), **fields,
              'messages': messages_to_ollama(model_messages)}
    original_context, original_messages = deepcopy(context), deepcopy(model_messages)
    # 使用正式_model_request路径，包含最终预算检查；不注入私有候选函数。
    fitted = json.loads(model_request(model_messages, **fields).data)
    assert context == original_context and model_messages == original_messages
    tokens = request_tokens(fitted)
    budget = fitted['options']['num_ctx'] - fitted['options']['num_predict']
    assert tokens <= budget
    result = {'case': 'F002:no_rules', 'original_estimated_tokens': request_tokens(before),
              'fitted_estimated_tokens': tokens, 'input_budget': budget,
              'context_and_native_messages_unchanged': True,
              'source_sha256': sha256((ROOT / 'src/agent/react_loop.py').read_bytes()).hexdigest(),
              'frozen_agent_sha256': sha256((HERE / 'agent.json').read_bytes()).hexdigest(),
              'request_only': True, 'actual_model_usage': None}
    (HERE / 'I01_正式补丁轨迹预算.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (HERE / 'I01_修复后实际请求.json').write_text(json.dumps(fitted, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    raise Captured

with patch.object(react_loop, '_model_request', capture), patch.object(react_loop, 'route_question', lambda *a, **k: None):
    try:
        list(react_loop._observe_events(question, get_available_tools(), context, messages, thought=state['thought']))
    except Captured:
        pass
