"""部署启动器：只运行当前原生应用，保留每次启动的独立日志。"""
from pathlib import Path
import os
import subprocess
import sys

project = Path(r'E:\工作\RAG-Agent')
run = project / '.test-tmp/win-audit-deploy-20261008'
name = sys.argv[1]
environment = {**os.environ, 'PYTHONIOENCODING':'utf-8', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'TOKENIZERS_PARALLELISM':'false', 'RAG_OLLAMA_BASE_URL':'http://127.0.0.1:11434'}
with (run / (name+'-stdout.log')).open('xb') as output, (run / (name+'-stderr.log')).open('xb') as error:
    code = subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'src/frontend/app.py', '--server.address=0.0.0.0', '--server.port=8501', '--browser.gatherUsageStats=false'], cwd=project, env=environment, stdout=output, stderr=error).returncode
raise SystemExit(code)
