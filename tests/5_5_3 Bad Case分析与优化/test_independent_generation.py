"""独立审计缺陷回归：移植原预期行为断言；模型报文和微型向量为明确替身。"""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from threading import Thread
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from src.generation.cache import SemanticCache
from src.generation.rag_pipeline import _finish_generation, build_context, generate_answer, resolve_citations
from src.generation.streaming import stream_answer
from src.utils.config import generation_options, load_config, ollama_base_url

def packets(*items):
    """仅模拟 NDJSON 传输，不模拟回答质量。"""
    return BytesIO(b'\n'.join((json.dumps(item, ensure_ascii=False).encode() for item in items)) + b'\n')

class GenerationAudit(unittest.TestCase):

    def setUp(self):
        self.config = deepcopy(load_config())
        for module in ('src.generation.rag_pipeline', 'src.generation.cache', 'src.frontend.components.documents'):
            mock = patch(module + '.load_config', return_value=self.config)
            mock.start()
            self.addCleanup(mock.stop)
        self.doc = Document(page_content='实验准确率为81%。第二句说明数据来源。', metadata={'source_file': '审计.txt', 'doc_id': 'a' * 64, 'chunk_id': '审计块', 'line_start': 1, 'line_end': 2})
        self.context = build_context('准确率是多少？', [(self.doc, 0.9)])
        self.done = {'model': 'qwen2.5:7b', 'done': True, 'done_reason': 'stop', 'message': {'content': '准确率为81%。[参考文档1]'}, 'prompt_eval_count': 20, 'eval_count': 10}
        self.result = {'type': 'done', **_finish_generation(self.done, self.context, generation_options(self.config['llm']))}

    def test_G02_indented_code_citation_must_not_validate_or_enter_cache(self):
        resolved = resolve_citations('仅展示引用格式，未给出事实答案：\n\n    [参考文档1]', self.context)
        with patch('src.generation.cache.get_embeddings') as embedding:
            embedding.return_value.embed_query.return_value = [1.0, 0.0]
            cached = SemanticCache().put('展示格式', {**self.result, **resolved}, '审计范围')
        print('G02 observed:', json.dumps({'citation_count': len(resolved['citations']), 'missing_citations': resolved['missing_citations'], 'warnings': resolved['warnings'], 'cached': cached}, ensure_ascii=False))
        self.assertEqual(resolved['citations'], [], '四空格缩进代码不能充当正文事实引用')
        self.assertFalse(cached)

    def test_G03_nonboolean_done_must_not_end_stream_successfully(self):
        malformed = {**self.done, 'done': 'false'}
        with patch('src.generation.streaming.urlopen', return_value=packets(malformed)):
            events = list(stream_answer('问题', self.context))
        print('G03 observed:', json.dumps({'done_value': malformed['done'], 'event_types': [e['type'] for e in events], 'final_reason': events[-1].get('done_reason')}, ensure_ascii=False))
        self.assertEqual(events[-1]['type'], 'error', '字符串 false 不是协议布尔完成标记')

    def test_claimed_quote_must_exist_in_its_cited_excerpt(self):
        """模型输出为替身，只检验引句定位及缓存拒绝，不冒充语义质量评测。"""
        raw = '原文依据："准确率为99%。"。[参考文档1]\n说明：准确率为99%。[参考文档1]'
        result = _finish_generation({**self.done, 'message': {'content': raw}}, self.context, {})
        self.assertEqual(result['evidence_quote_errors'], [[1]])
        self.assertNotIn('99%', result['answer'])
        self.assertEqual(result['raw_answer'], raw)
        self.assertFalse(SemanticCache().put('准确率', {'type': 'done', **result}, '范围'))

    def test_whitespace_in_original_quote_is_preserved_as_evidence(self):
        raw = '原文依据："实验准确率为81%。\n第二句说明数据来源。"。[参考文档1]'
        result = _finish_generation({**self.done, 'message': {'content': raw}}, self.context, {})
        self.assertEqual(result['evidence_quote_errors'], [])
        self.assertIn('81%', result['answer'])

    def test_pdf_line_break_and_ligature_must_not_reject_real_quote(self):
        """只消除PDF排版差异，不能改写事实数字或匹配其他来源。"""
        context = deepcopy(self.context)
        context['references'][0]['text'] = 'we improve efﬁciency for box pre-\ndiction.'
        raw = '原文依据："We improve efficiency for box prediction."。[参考文档1]'
        result = _finish_generation({**self.done, 'message': {'content': raw}}, context, {})
        self.assertEqual(result['evidence_quote_errors'], [])

    def test_literal_hyphen_and_changed_number_remain_distinct(self):
        context = deepcopy(self.context)
        context['references'][0]['text'] = 'end-to-end accuracy is 81%.'
        for quote in ('endtoend accuracy is 81%.', 'end-to-end accuracy is 99%.'):
            with self.subTest(quote=quote):
                raw = f'原文依据："{quote}"。[参考文档1]'
                result = _finish_generation({**self.done, 'message': {'content': raw}}, context, {})
                self.assertEqual(result['evidence_quote_errors'], [[1]])

    def test_quote_from_another_reference_does_not_validate(self):
        context = deepcopy(self.context)
        context['references'].append({**context['references'][0], 'id': 2, 'text': '训练数据为甲数据集。'})
        raw = '原文依据："训练数据为甲数据集。"。[参考文档1]'
        result = _finish_generation({**self.done, 'message': {'content': raw}}, context, {})
        self.assertEqual(result['evidence_quote_errors'], [[1]])

    def test_nonstream_validation_failure_preserves_returned_usage(self):
        malformed = {**self.done, 'message': {'content': ''}}
        with patch('src.generation.rag_pipeline.urlopen', return_value=BytesIO(json.dumps(malformed).encode())):
            with self.assertRaises(RuntimeError) as caught:
                generate_answer('问题', self.context)
        self.assertEqual(caught.exception.usage, {'prompt_eval_count': 20, 'eval_count': 10})

    def test_stream_validation_failure_preserves_returned_usage(self):
        with patch('src.generation.streaming.urlopen', return_value=packets({**self.done, 'done': 'false'})):
            events = list(stream_answer('问题', self.context))
        self.assertEqual(events[-1]['type'], 'error')
        self.assertEqual(events[-1]['usage'], {'prompt_eval_count': 20, 'eval_count': 10})

    def test_G01_redirect_must_not_bypass_local_host_allowlist(self):
        """两个端口都在127.0.0.1；127.1是同一本机的缩写，但不在配置允许名单。"""
        observed = []
        payload = json.dumps(self.done, ensure_ascii=False).encode()

        class Target(BaseHTTPRequestHandler):

            def do_GET(self):
                observed.append({'method': 'GET', 'host': self.headers.get('Host'), 'path': self.path})
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        @contextmanager
        def server(handler):
            instance = ThreadingHTTPServer(('127.0.0.1', 0), handler)
            thread = Thread(target=instance.serve_forever, daemon=True)
            thread.start()
            try:
                yield instance.server_port
            finally:
                instance.shutdown()
                instance.server_close()
                thread.join(timeout=2)
        with server(Target) as target_port:
            redirect_url = f'http://127.1:{target_port}/redirected-response'
            with self.assertRaisesRegex(ValueError, '只允许本机'):
                ollama_base_url({**self.config['llm'], 'base_url': redirect_url})

            class Redirect(BaseHTTPRequestHandler):

                def do_POST(self):
                    self.rfile.read(int(self.headers.get('Content-Length', '0')))
                    self.send_response(302)
                    self.send_header('Location', redirect_url)
                    self.end_headers()

                def log_message(self, *_args):
                    pass
            with server(Redirect) as source_port:
                self.config['llm']['base_url'] = f'http://127.0.0.1:{source_port}'
                with self.assertRaises(RuntimeError):
                    generate_answer('问题', self.context)
        print('G01 observed:', json.dumps({'redirect_target_requests': observed}, ensure_ascii=False))
        self.assertEqual(observed, [], '重定向目标也应通过本机允许名单，或禁用重定向')

if __name__ == "__main__":
    unittest.main()
