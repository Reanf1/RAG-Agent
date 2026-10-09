"""本轮 Windows 启动器：按需任务持有进程，各次日志独立保存。"""
from pathlib import Path
import os
import subprocess
import sys

project = Path(r"E:\工作\RAG-Agent")
run = project / ".test-tmp/win-simplify-20261009"
mode = sys.argv[1]
environment = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   RAG_OLLAMA_BASE_URL="http://127.0.0.1:11434")
if mode == "ollama":
    environment.update(OLLAMA_HOST="127.0.0.1:11434", OLLAMA_NO_CLOUD="1",
                       OLLAMA_NUM_PARALLEL="1")
    command = [str(Path(os.environ["LOCALAPPDATA"]) / "Programs/Ollama/ollama.exe"), "serve"]
elif mode.startswith("regression"):
    label = "regression-initial" if mode == "regression" else mode
    command = [sys.executable, "reports/模块完整性验证/verify_completeness_tests.py",
               "--scope", "all", "--output", str(run / (label + ".json"))]
else:
    command = [sys.executable, "-m", "streamlit", "run", "src/frontend/app.py",
               "--server.address=0.0.0.0", "--server.port=8501",
               "--browser.gatherUsageStats=false"]
with (run / (mode + "-stdout.log")).open("xb") as output, (run / (mode + "-stderr.log")).open("xb") as error:
    code = subprocess.run(command, cwd=project, env=environment, stdout=output, stderr=error).returncode
raise SystemExit(code)
