"""会话内有上限的答案缓存：精确匹配优先，本地 M3E 余弦相似度匹配其次。"""

from copy import deepcopy
import hashlib
import json
import math
import re
from threading import Lock

from src.generation.prompt_template import PROMPT_VERSION, build_rag_messages
from src.retrieval.vector_store import VectorStore, get_embeddings
from src.utils.config import load_config


def cache_scope(store: VectorStore) -> str:
    """正文/来源和生成配置共同确定有效范围，不用块数冒充知识库版本。

    本科规模每次读取当前小型语料并计算哈希，新增、删除及同数量替换均失效。
    不修改索引，也不加载模型；排序排除数据库返回顺序变化的影响。
    """
    config = load_config()
    corpus = sorted(json.dumps({"text": doc.page_content, "metadata": doc.metadata},
                               ensure_ascii=False, sort_keys=True) for doc in store.list_chunks())
    payload = {"prompt_version": PROMPT_VERSION,
               "prompt": [(message.type, message.content) for message in build_rag_messages("缓存模板校验", "缓存上下文")],
               "settings": {key: config[key] for key in ("llm", "embedding", "retrieval", "generation")},
               "index_path": config["paths"]["vector_index"], "corpus": corpus}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _constraints(question: str) -> tuple:
    """保守过滤容易被向量相似度忽略的变化；这是规则，不是语义等价证明。"""
    # 正负号和百分／千分符号属于数值条件，不能只保留数字部分。
    numbers = tuple(re.sub(r"\s+", "", value) for value in
                    re.findall(r"[+-]?\d+(?:\.\d+)?(?:\s*[%‰])?", question))
    chinese_numbers = tuple(re.findall(r"[零〇一二两三四五六七八九十百千万亿]+(?:点[零〇一二三四五六七八九]+)?", question))
    # 否定对象无法只用有／无否定的布尔值表达；不同改写保守地重新检索。
    negated = bool(re.search(r"不|没|无|非|\b(?:not|no|without|never)\b|n't\b", question, re.I))
    negated_question = re.sub(r"\s+", "", question.casefold()) if negated else ""
    # 相反的比较方向常有极高向量相似度；保守保留原比较词，不猜同义条件。
    comparisons = tuple(match.lower() for match in re.findall(
        r">=|<=|≥|≤|>|<|大于等于|小于等于|不超过|不低于|不高于|不少于|不多于|"
        r"至少|至多|最多|最少|大于|小于|超过|低于|高于|"
        r"\b(?:at least|at most|more than|less than|greater than|lower than|higher than)\b", question, re.I))
    return (numbers, chinese_numbers, comparisons,
            tuple(sorted(set(re.findall(r"(?<![A-Za-z0-9])[A-Z]{2,}[A-Za-z0-9_-]*|(?:论文|文献)[A-Za-z0-9_-]+", question)))),
            negated_question,
            bool(re.search(r"[\u4e00-\u9fff]", question)),
            bool(re.search(r"中文|汉语|Chinese", question, re.I)),
            bool(re.search(r"英文|英语|English", question, re.I)))


class SemanticCache:
    """实例放入 Streamlit session_state，用户会话之间不共享答案或问题。"""

    def __init__(self):
        self.settings = deepcopy(load_config()["generation"]["cache"])
        threshold, capacity = self.settings["similarity_threshold"], self.settings["max_entries"]
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("缓存相似度阈值必须在 (0, 1] 内")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("缓存容量必须为正整数")
        self.entries, self.scope = [], None
        # 同一Agent并行批次可能同时访问本会话缓存，读写短区间由工具入口加锁。
        self.lock = Lock()

    def clear(self):
        """清空答案/问题向量；不影响知识库及原文件。"""
        self.entries.clear()
        self.scope = None

    def _bind(self, scope):
        if self.scope != scope:
            self.clear()
            self.scope = scope

    def _vector(self, question):
        # M3E 上限512 Token；长问题只精确匹配，避免截断尾部后误判为相同问题。
        if len(question) > 256:
            return None
        vector = get_embeddings().embed_query(question)
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("问题向量必须非空、有限且非零")
        return tuple(value / norm for value in vector)

    def lookup(self, question: str, scope: str) -> dict | None:
        """命中返回独立答案快照，否则返回None，由调用方重新检索。

        顺序为范围校验→问题精确匹配→关键约束一致→向量相似度匹配。
        精确匹配无需编码；语义匹配在单位向量上计算点积，即余弦相似度。
        返回值深拷贝后将本次LLM用量置零，原始用量另存，不修改缓存条目。
        """
        question = question.strip()
        if not question:
            raise ValueError("缓存问题不能为空")
        self._bind(scope)
        best, similarity, mode = None, -1.0, "semantic"
        for entry in self.entries:
            if entry["question"] == question:
                best, similarity, mode = entry, 1.0, "exact"
                break
        if best is None:
            # 本次问题的数字、实体和否定约束不随条目改变，只解析一次。
            constraints = _constraints(question)
            eligible = [entry for entry in self.entries if entry["vector"] is not None
                        and _constraints(entry["question"]) == constraints]
            if not eligible or len(question) > 256:
                return None
            vector = self._vector(question)
            for entry in eligible:
                if len(vector) != len(entry["vector"]):
                    raise ValueError("缓存向量维度不一致，请清空缓存并检查模型配置")
                score = max(-1.0, min(1.0, sum(a * b for a, b in zip(vector, entry["vector"]))))
                if score >= self.settings["similarity_threshold"] and score > similarity:
                    best, similarity = entry, score
        if best is None:
            return None
        result = deepcopy(best["result"])
        result["original_usage"] = result["usage"]
        result["usage"] = {key: 0 for key in result["usage"]}
        result["cache"] = {"hit": True, "mode": mode, "similarity": similarity,
                           "question": best["question"]}
        return result

    def put(self, question: str, result: dict, scope: str) -> bool:
        """只存正常结束、非空且有有效来源的回答；最早写入的超额条目淘汰。"""
        self._bind(scope)
        question = question.strip()
        if not question or result.get("type") != "done" or result.get("done_reason") != "stop" \
                or result.get("generation_mode") in {"empty", "low"} \
                or not result.get("answer", "").strip() or not result.get("citations") \
                or result.get("warnings") or result.get("invalid_citation_ids") or result.get("missing_citations"):
            return False
        vector = self._vector(question)
        self.entries = [entry for entry in self.entries if entry["question"] != question]
        self.entries.append({"question": question, "vector": vector, "result": deepcopy(result)})
        self.entries = self.entries[-self.settings["max_entries"]:]
        return True
