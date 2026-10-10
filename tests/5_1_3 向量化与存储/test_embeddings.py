"""5.1.3 向量化与存储：TestLocalEmbeddings、TestEmbeddingEvaluation。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tempfile
import unittest
from unittest.mock import patch
from src.retrieval.vector_store import get_embeddings


class TestLocalEmbeddings(unittest.TestCase):
    """隔离大模型，验证配置、本地约束和延迟加载缓存。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        get_embeddings.cache_clear()
        self.addCleanup(get_embeddings.cache_clear)
        self.config = {"embedding": {
            "local_path": self.directory.name, "device": "cpu", "batch_size": 32,
        }}
        patcher = patch("src.retrieval.vector_store.load_config", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_local_loading_and_normalization_arguments(self):
        """只从本地目录加载，关闭远程代码，文档编码开启归一化。"""
        with patch("src.retrieval.vector_store.HuggingFaceEmbeddings") as model:
            self.assertIs(get_embeddings(), model.return_value)
            model.assert_called_once_with(
                model_name=self.directory.name,
                model_kwargs={"device": "cpu", "local_files_only": True, "trust_remote_code": False},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )


    def test_parallel_first_load_constructs_only_once(self):
        """三线程同时首次调用，构造器只执行一次；失败不锁死后续请求。"""
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        from time import sleep
        gate, instance = Barrier(3), object()
        def construct(*args, **kwargs):
            sleep(.08)
            return instance
        def request():
            gate.wait(timeout=2)
            return get_embeddings()
        with patch("src.retrieval.vector_store.HuggingFaceEmbeddings", side_effect=construct) as model, ThreadPoolExecutor(3) as pool:
            futures = [pool.submit(request) for _ in range(3)]
            self.assertTrue(all(future.result(timeout=3) is instance for future in futures))
            self.assertEqual(model.call_count, 1)


if __name__ == "__main__":
    unittest.main()
