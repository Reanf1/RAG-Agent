"""跨小节共用的明确构造样例；不作为真实模型质量或性能证据。"""

from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO
import json
import tempfile
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from src.utils.config import load_config


class SmallEmbeddings(Embeddings):
    """测试使用明确的二维向量，不用于实验效果或速度结论。"""

    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def _vector(self, text):
        if "农业" in text:
            return [0.0, 1.0]
        if "反向" in text:
            return [-1.0, 0.0]
        return [1.0, 0.0]

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return self._vector(text)


class StreamingResponse(BytesIO):
    """按真实 NDJSON 行迭代，跟踪读包与关闭，不能用作模型性能证据。"""

    def __init__(self, packets):
        super().__init__(b"".join(json.dumps(packet, ensure_ascii=False).encode() + b"\n" for packet in packets))
        self.read_packets = 0

    def __next__(self):
        line = super().__next__()
        self.read_packets += 1
        return line


@contextmanager
def isolated_agent_logs():
    """每个 Agent 测试模块使用临时日志，避免模拟请求写入业务日志。"""
    with tempfile.TemporaryDirectory(prefix="agent-metrics-tests-") as directory:
        config = deepcopy(load_config())
        config["paths"]["logs"] = directory
        # 上下文式 patch 不会被工具测试的 patch.stopall() 撤销。
        with patch("src.utils.logger.load_config", return_value=config):
            yield
