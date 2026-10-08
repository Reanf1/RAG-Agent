"""按原审计真实表格坐标核验六项缺陷与两项对照，避免误选同页其他表。"""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from verify_real_repairs import save
from src.data_loader.pdf_loader import load_pdf

audit = Path('/Users/rean/github/测试项')
original = json.loads((audit / 'docs/完整性测试证据/真实独立表标题逐张审计.json').read_text())['tables']
cases = [row for row in original if 'visual_review' in row]
results = []
for row in cases:
    path = audit / 'data/raw/evaluation_vision_transformers' / row['source_file']
    tables = [doc for doc in load_pdf(path) if doc.metadata.get('content_type') == 'table']
    matches = [doc for doc in tables if any(region['page_number'] == row['page_number']
               and all(abs(a - b) < .1 for a, b in zip(region['bbox'], row['bbox']))
               for region in json.loads(doc.metadata['table_regions']))]
    # 排除样例原附标题已正确；真实错配样例以原页核对确认的标题为预期。
    expected = row['below_caption'] if row['visual_review']['confirmed_mismatch'] else row['attached_first_line']
    actual = [doc.page_content.splitlines()[0] for doc in matches]
    results.append({'file': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'page_number': row['page_number'], 'bbox': row['bbox'], 'expected': expected,
                    'actual': actual, 'passed': actual == [expected], 'confirmed_defect': row['visual_review']['confirmed_mismatch']})
save(Path(sys.argv[1]), results)
print(json.dumps({'total': len(results), 'passed': sum(item['passed'] for item in results)}, ensure_ascii=False))
assert len(results) == 8 and all(item['passed'] for item in results)
