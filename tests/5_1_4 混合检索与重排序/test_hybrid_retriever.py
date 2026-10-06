"""5.1.4 混合检索与重排序：TestRRFFusion、TestHybridRetriever。"""

import sys
from pathlib import Path

# 从其他目录直接运行测试文件时，也能定位 src 和共用样例。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tempfile
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever, rrf_fusion
from tests.helpers import SmallEmbeddings


class TestRRFFusion(unittest.TestCase):
    """用明确排名和手算分数核验融合，不用原始分数冒充排名贡献。"""

    def setUp(self):
        self.documents = {key: Document(page_content=f"论文 {key}", metadata={
            "chunk_id": key, "doc_id": key, "source_file": f"{key}.pdf", "page_number": 1})
            for key in ("a", "b", "c", "d")}

    def test_one_based_rank_formula_and_shared_chunk_bonus(self):
        """两路共同命中的块分数累加，第一名贡献为 1/61。"""
        d = self.documents
        found = rrf_fusion([(d["a"], 0.9), (d["b"], 0.8), (d["c"], -0.2)],
                           [(d["b"], 8.0), (d["c"], 2.0), (d["d"], 0.0)])
        expected = {"a": 1 / 61, "b": 1 / 62 + 1 / 61,
                    "c": 1 / 63 + 1 / 62, "d": 1 / 63}
        self.assertEqual([doc.metadata["chunk_id"] for doc, _ in found], ["b", "c", "a", "d"])
        for document, score in found:
            self.assertAlmostEqual(score, expected[document.metadata["chunk_id"]])
            self.assertIs(document, d[document.metadata["chunk_id"]])

    def test_raw_score_scale_does_not_affect_fusion(self):
        """负分、零分与大数的原始值都不参与 RRF，排名相同则结果相同。"""
        d = self.documents
        original = rrf_fusion([(d["a"], 0.9), (d["b"], 0.2)], [(d["b"], 5), (d["c"], 0)])
        changed = rrf_fusion([(d["a"], 999), (d["b"], -1)], [(d["b"], -200), (d["c"], -800)])
        self.assertEqual(original, changed)

    def test_duplicate_within_one_route_counts_only_once(self):
        """单路重复不多次加分，保留首次出现的位置，两路贡献仍可叠加。"""
        d = self.documents
        found = rrf_fusion([(d["a"], 1), (d["a"], 1), (d["b"], 0.5)],
                           [(d["a"], 2), (d["a"], 2)])
        self.assertEqual(len(found), 2)
        self.assertAlmostEqual(found[0][1], 2 / 61)
        self.assertAlmostEqual(found[1][1], 1 / 63)

    def test_same_content_in_different_sources_is_not_merged(self):
        """合并键是 chunk_id，正文相同也保留不同文件与页码。"""
        a = Document(page_content="相同正文", metadata={"chunk_id": "a", "doc_id": "a",
                     "source_file": "A.pdf", "page_number": 2, "page_end": 3})
        b = Document(page_content="相同正文", metadata={"chunk_id": "b", "doc_id": "b",
                     "source_file": "B.docx", "table_index": 1})
        self.assertEqual([doc for doc, _ in rrf_fusion([(a, 1)], [(b, 2)])], [a, b])

    def test_equal_scores_have_stable_chunk_id_order(self):
        """两路交换同分候选的输入顺序，仍按稳定块 ID 打破平分。"""
        d = self.documents
        first = rrf_fusion([(d["b"], 1)], [(d["a"], 2)])
        second = rrf_fusion([(d["a"], 2)], [(d["b"], 1)])
        self.assertEqual(first, second)
        self.assertEqual([doc.metadata["chunk_id"] for doc, _ in first], ["a", "b"])

    def test_empty_routes_and_zero_smoothing(self):
        """单路为空时只使用另一支，双空返回空；k=0 时分母从 1 开始。"""
        d = self.documents
        route = [(d["a"], -1), (d["b"], 0)]
        self.assertEqual(rrf_fusion([], []), [])
        self.assertEqual(rrf_fusion(route, []), rrf_fusion([], route))
        self.assertEqual([score for _, score in rrf_fusion(route, [], rrf_k=0)], [1, 0.5])

    def test_missing_ids_and_invalid_smoothing_are_rejected(self):
        """身份缺失不能退回按正文合并；平滑常数的错误输入明确报错。"""
        for value in (None, "", "  ", 1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "chunk_id"):
                rrf_fusion([(Document(page_content="正文", metadata={"chunk_id": value}), 1)], [])
        for value in (-1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "rrf_k"):
                rrf_fusion([], [], rrf_k=value)


class TestHybridRetriever(unittest.TestCase):
    """真实 Chroma/BM25 联调，两维向量仅用于确定行为与模型调用次数。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.embeddings = SmallEmbeddings()
        self.store = VectorStore(self.directory.name, self.embeddings)
        self.chunks = [Document(page_content=text, metadata={
            "chunk_id": str(i), "doc_id": str(i), "source_file": f"论文{i}.pdf", "page_number": i + 1})
            for i, text in enumerate(("神经网络研究", "BatchNormalization BatchNormalization",
                                     "BatchNormalization training training training", "农业数据", "反向向量"))]
        self.store.add_chunks(self.chunks)
        self.retriever = HybridRetriever(self.store)

    def test_both_routes_take_candidates_before_final_top_k(self):
        """最终只取 1 个也先各召回 20 个候选，不提前把两路截为 Top-1。"""
        original_bm25_search = BM25Retriever.search
        with (patch.object(self.store, "search", wraps=self.store.search) as vector,
              patch.object(BM25Retriever, "search", autospec=True, side_effect=original_bm25_search) as bm25):
            found = self.retriever.search("BatchNormalization", k=1)
        vector.assert_called_once_with("BatchNormalization", k=20, doc_id=None)
        self.assertEqual(bm25.call_args.kwargs, {"k": 20, "doc_id": None})
        self.assertIs(bm25.call_args.args[0].vector_store, self.store)
        self.assertEqual(len(found), 1)
        self.assertIn(found[0][0], self.chunks[1:3])
        self.assertGreater(found[0][1], 1 / 61)
        self.assertEqual(self.embeddings.query_calls, ["BatchNormalization"])
        self.assertEqual(self.embeddings.document_calls, [[doc.page_content for doc in self.chunks]])

    def test_image_markers_do_not_compete_with_actual_paper_evidence(self):
        """复现W08：图片区的占位描述曾被模型排在1D位置编码原文之前。"""
        image = Document(page_content="[图像区域 213：原文第 21 页]\n121\n122", metadata={"chunk_id": "image"})
        text = Document(page_content="We use standard learnable 1D position embeddings.", metadata={"chunk_id": "text"})
        with patch.object(self.store, "search", return_value=[(image, 0.9), (text, 0.8)]), \
                patch.object(BM25Retriever, "search", return_value=[(image, 1.0)]), \
                patch("src.retrieval.reranker.get_reranker") as model:
            model.return_value.predict.side_effect = lambda pairs, **kw: [0.5] * len(pairs)
            found = self.retriever.search("1D还是2D？", k=1, rerank=True)
        self.assertEqual(found[0][0], text)
        pairs = model.return_value.predict.call_args.args[0]
        self.assertEqual([pair[1] for pair in pairs], [text.page_content])

    def test_document_filter_and_empty_bm25_route(self):
        """两路共同过滤文档，BM25 无匹配时保留向量路的排名贡献。"""
        found = self.retriever.search("BatchNormalization", doc_id="3")
        self.assertEqual([doc for doc, _ in found], [self.chunks[3]])
        self.assertAlmostEqual(found[0][1], 1 / 61)
        before = len(self.embeddings.query_calls)
        self.assertEqual(self.retriever.search("BatchNormalization", doc_id="missing"), [])
        self.assertEqual(len(self.embeddings.query_calls), before)

    def test_blank_query_and_empty_store_do_not_load_model(self):
        """空问题不调用两路，空库不会初始化 Embedding。"""
        with patch.object(self.store, "search", side_effect=AssertionError("不应查询")):
            self.assertEqual(self.retriever.search(" \n"), [])
        with patch("src.retrieval.vector_store.get_embeddings", side_effect=AssertionError("不应加载模型")):
            empty = HybridRetriever(VectorStore(Path(self.directory.name) / "empty"))
            self.assertEqual(empty.search("BERT"), [])

    def test_reused_retriever_reads_current_corpus_after_add_and_delete(self):
        """同一个混合检索实例也读取最新正文，新增和删除后无陈旧 BM25 块。"""
        new = Document(page_content="Reranker", metadata={"chunk_id": "new", "doc_id": "new"})
        self.store.add_chunks([new])
        self.assertEqual(self.retriever.search("Reranker", k=1)[0][0], new)
        self.store.delete_document("new")
        found = self.retriever.search("Reranker", k=20)
        self.assertEqual({doc.metadata["chunk_id"] for doc, _ in found}, {str(i) for i in range(5)})
        self.assertEqual(len(self.embeddings.document_calls), 2)

    def test_default_k_large_k_and_yaml_smoothing(self):
        """默认 K 与存储一致；更大的 K 扩大召回，平滑常数从 YAML 读取。"""
        from src.utils.config import load_config
        config = load_config()
        config["retrieval"]["rrf_k"] = 0
        self.store.top_k = 2
        with patch("src.retrieval.hybrid_retriever.load_config", return_value=config):
            retriever = HybridRetriever(self.store)
        self.assertEqual(len(retriever.search("unknownkeyword")), 2)
        self.assertEqual([score for _, score in retriever.search("unknownkeyword")], [1, 0.5])
        extra = [Document(page_content="BatchNormalization", metadata={"chunk_id": f"extra-{i}", "doc_id": "extra"})
                 for i in range(41)]
        self.store.add_chunks(extra)
        found = retriever.search("BatchNormalization", k=100)
        self.assertEqual(len(found), 46)
        self.assertEqual(len({doc.metadata["chunk_id"] for doc, _ in found}), 46)

    def test_invalid_parameters_and_route_errors_do_not_silently_degrade(self):
        """参数错误直接拒绝，两路异常明确上报，不能以半成品冒充混合检索。"""
        from src.utils.config import load_config
        for value in (0, -1, True, 1.5):
            with self.subTest(k=value), self.assertRaises(ValueError):
                self.retriever.search("BERT", k=value)
        for key, value in (("candidate_k", 0), ("candidate_k", True), ("rrf_k", -1)):
            config = load_config()
            config["retrieval"][key] = value
            with self.subTest(key=key), patch("src.retrieval.hybrid_retriever.load_config", return_value=config):
                with self.assertRaisesRegex(ValueError, key):
                    HybridRetriever(self.store)
        with patch.object(self.store, "search", side_effect=RuntimeError("向量查询失败")):
            with self.assertRaisesRegex(RuntimeError, "向量查询失败"):
                self.retriever.search("BERT")
        with patch.object(BM25Retriever, "search", side_effect=RuntimeError("BM25 查询失败")):
            with self.assertRaisesRegex(RuntimeError, "BM25 查询失败"):
                self.retriever.search("BERT")

    def test_model_reranks_fused_candidates_before_final_top_k(self):
        """模型先看到融合 Top-20；候选末项也能经模型升到最终 Top-1。"""
        self.store.add_chunks([Document(page_content=f"论文候选 {i}", metadata={
            "chunk_id": f"extra-{i}", "doc_id": "extra"}) for i in range(25)])
        candidates = self.retriever.search("unknownkeyword", k=20)
        with patch("src.retrieval.reranker.get_reranker") as model:
            model.return_value.predict.return_value = [0.01] * 19 + [0.99]
            found = self.retriever.search("unknownkeyword", k=1, rerank=True)
            self.assertEqual(found, [(candidates[-1][0], 0.99)])
            model.return_value.predict.assert_called_once_with(
                [["unknownkeyword", doc.page_content] for doc, _ in candidates],
                batch_size=8, show_progress_bar=False)
            # K 大于默认候选数时相应扩大，而不是返回不足 K 的人为截断结果。
            model.return_value.predict.return_value = [0.5] * 25
            found = self.retriever.search("unknownkeyword", k=25, rerank=True)
            self.assertEqual(len(found), 25)
        self.assertEqual(len(self.embeddings.document_calls), 2)

    def test_model_reranking_respects_filter_and_empty_results(self):
        """过滤先应用于两路，单条候选也精排；空结果和空问题不加载模型。"""
        with patch("src.retrieval.reranker.get_reranker") as model:
            model.return_value.predict.return_value = [0.7]
            found = self.retriever.search("BatchNormalization", doc_id="3", rerank=True)
            self.assertEqual(found, [(self.chunks[3], 0.7)])
            model.return_value.predict.assert_called_once_with(
                [["BatchNormalization", self.chunks[3].page_content]], batch_size=8, show_progress_bar=False)
            model.reset_mock()
            self.assertEqual(self.retriever.search("BatchNormalization", doc_id="missing", rerank=True), [])
            self.assertEqual(self.retriever.search("  ", rerank=True), [])
            model.assert_not_called()

    def test_output_language_and_citation_suffix_does_not_change_retrieval(self):
        """同一科研问题附加输出要求时，三路使用同一内容问题，避免核心证据掉出候选。"""
        question = "ViT如何使用位置编码？"
        with patch.object(self.store, "search", wraps=self.store.search) as vector, \
                patch.object(BM25Retriever, "search", autospec=True, wraps=BM25Retriever.search) as bm25, \
                patch("src.retrieval.reranker.get_reranker") as model:
            bm25.return_value = []
            model.return_value.predict.side_effect = lambda pairs, **kw: [.5] * len(pairs)
            self.retriever.search(question + "请用中文简述并注明原文页码。", doc_id="3", rerank=True)
        vector.assert_called_once_with(question, k=20, doc_id="3")
        self.assertEqual(bm25.call_args.args[1], question)
        self.assertEqual(model.return_value.predict.call_args.args[0][0][0],
                         question + "\npositional position encoding embeddings")

    def test_output_suffix_cleanup_preserves_content_questions(self):
        """不删除实体、数字、否定、多问句或语言本身作为研究对象的文字。"""
        questions = ["中文回答与英文回答的准确率有什么差异？", "请用中文回答是什么意思？",
                     "ViT是否不用2D位置编码？它的实验结果是什么？"]
        with patch.object(self.store, "search", return_value=[]) as vector:
            for question in questions:
                with self.subTest(question=question):
                    self.retriever.search(question)
                    self.assertEqual(vector.call_args.args[0], question)
            self.retriever.search(questions[-1] + "请注明来源。")
            self.assertEqual(vector.call_args.args[0], questions[-1])

    def test_plain_rrf_does_not_load_model_and_model_failure_is_reported(self):
        """不启用精排时不依赖重排权重；启用后失败不得返回未经重排的候选。"""
        with patch("src.retrieval.reranker.get_reranker", side_effect=RuntimeError("重排推理失败")) as model:
            self.assertTrue(self.retriever.search("BERT"))
            model.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "重排推理失败"):
                self.retriever.search("BERT", rerank=True)


if __name__ == "__main__":
    unittest.main()
