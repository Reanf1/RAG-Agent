# 与课程开发环境一致；模型和用户数据通过挂载保存，不打进镜像。
FROM python:3.10.10-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    TZ=Asia/Shanghai

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Python 3.10.10 基础镜像自带 SQLite 3.34.1，低于 Chroma 的要求。
# 仅在 Linux amd64 镜像替换 SQLite 驱动，保留 Windows/Mac 原生环境。
RUN python -m pip install --no-cache-dir pysqlite3-binary==0.5.4 \
    && python -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'rag_sqlite.pth').write_text(\"import sys, pysqlite3; sys.modules['sqlite3'] = pysqlite3\\n\", encoding='utf-8')"
# 独立新进程核验启动钩子，构建时就暴露 SQLite/Chroma 不兼容。
RUN python -c "import sqlite3, chromadb; assert sqlite3.sqlite_version_info >= (3, 35, 0); print('SQLite', sqlite3.sqlite_version, 'Chroma', chromadb.__version__)"

COPY config.yaml ./
COPY src ./src

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["python", "-m", "streamlit", "run", "src/frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
