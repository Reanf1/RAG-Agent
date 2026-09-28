# 当前仅部署初始化页面；业务模块接入后再加入对应运行依赖。
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY config.yaml ./
COPY src ./src

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "src/frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
