"""5.1.4 混合检索与重排序：TestReranker、TestLocalReranker。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from src.retrieval.reranker import Reranker


class TestReranker(unittest.TestCase):
    """大模型隔离，核验成对输入、排序与来源，不将模拟分数当实验效果。"""

    def setUp(self):
        from src.utils.config import load_config
        self.config = load_config()
        self.config["retrieval"]["top_k"] = 2
        self.config["retrieval"]["reranker_batch_size"] = 2
        patcher = patch("src.retrieval.reranker.load_config", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.reranker = Reranker()
        self.candidates = [(Document(page_content=text, metadata={
            "chunk_id": str(i), "source_file": f"论文{i}.pdf", "page_number": i + 1}), 10 - i)
            for i, text in enumerate(("无关材料", "注意力机制", "Attention mechanism"))]

    def test_pairs_model_scores_order_and_sources(self):
        """只用模型分数精排，同分保持原候选顺序，正文和来源不修改。"""
        original = [(doc.page_content, dict(doc.metadata)) for doc, _ in self.candidates]
        with patch("src.retrieval.reranker.get_reranker") as model:
            model.return_value.predict.return_value = [0.1, 0.9, 0.9]
            found = self.reranker.rerank("注意力是什么？", self.candidates)
            model.return_value.predict.assert_called_once_with(
                [["注意力是什么？", doc.page_content] for doc, _ in self.candidates],
                batch_size=2, show_progress_bar=False)
        self.assertEqual(found, [(self.candidates[1][0], 0.9), (self.candidates[2][0], 0.9)])
        self.assertIs(found[0][0], self.candidates[1][0])
        self.assertEqual(original, [(doc.page_content, doc.metadata) for doc, _ in self.candidates])


    def test_inference_error_is_not_silently_replaced_by_rrf(self):
        with patch("src.retrieval.reranker.get_reranker") as model:
            model.return_value.predict.side_effect = RuntimeError("推理失败")
            with self.assertRaisesRegex(RuntimeError, "推理失败"):
                self.reranker.rerank("query", self.candidates)


if __name__ == "__main__":
    unittest.main()
