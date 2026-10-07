"""最终同步后仅核对文件、健康与Git状态；不修改索引或原文。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import requests

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists(): raise FileExistsError('不覆盖同步后核验记录')
results=args.project/'reports/5_5_2 系统性能评估/Windows正式性能评测_20261007'
source=json.loads((results/'I01_Windows源码核对.json').read_text(encoding='utf-8'))['source_sha256']
tests=json.loads((results/'I01_UTF8测试实际源码.json').read_text(encoding='utf-8'))
checks=[]
for name,expected in {**source,**tests}.items():
    raw=(args.project/name).read_bytes()
    # 只把Git的CRLF工作区转换为LF比较，保留实际字节指纹。
    checks.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'lf_sha256':hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest(),'expected_sha256':expected,'passed':hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==expected})
artifacts=json.loads((args.project/'reports/5_5_4 项目交付/最终交付核验_20261007.json').read_text(encoding='utf-8'))['artifact_sha256']
artifact_checks=[{'path':name,'passed':hashlib.sha256((args.project/name).read_bytes()).hexdigest()==expected} for name,expected in artifacts.items()]
app=requests.get('http://127.0.0.1:8501/_stcore/health',timeout=15)
model=requests.get('http://127.0.0.1:11434/api/version',timeout=15)
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.project,text=True).strip()
status=subprocess.check_output(['git','-c','core.quotepath=false','status','--short'],cwd=args.project,encoding='utf-8').splitlines()
report={'status':'passed' if all(r['passed'] for r in checks+artifact_checks) and app.status_code==200 and app.text.strip()=='ok' and model.status_code==200 else 'failed','revision':revision,'source_files':len(source),'test_files':len(tests),'file_checks':checks,'artifact_checks':artifact_checks,'app_health':{'http_status':app.status_code,'body':app.text.strip()},'ollama_health':{'http_status':model.status_code,'version':model.json()},'git_status':status,'boundary':'源码、测试文件与交付二进制核对；健康不代替模型推理，780项回归和真实推理使用已归档同源证据。'}
args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'revision':revision,'source_files':len(source),'test_files':len(tests),'artifacts':len(artifact_checks),'app_health':report['app_health'],'ollama_health':report['ollama_health'],'git_status':status},ensure_ascii=False))
assert report['status']=='passed'
