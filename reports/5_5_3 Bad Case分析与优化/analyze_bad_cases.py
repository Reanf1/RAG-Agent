"""从冻结实测提取Bad Case；结构检查与原文逐例分析分开，不代填人工分数。"""
import importlib.util
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "reports/5_5_2 系统性能评估"
spec = importlib.util.spec_from_file_location("evaluation", BASE / "evaluate_system.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def structural(rows, identifiers):
    """结构正确不代表事实、语义引用正确；分母明确列出。"""
    eligible, retained, invalid, failed = [], [], [], []
    for row in rows:
        results = [e for e in row["metrics"]["trace"] if e["type"] == "tool_result"]
        for event in results:
            if event["status"] == "error":
                failed.append({"id": row["id"], "name": event["name"], "args": event["args"], "error": event.get("error")})
        for call in row["tool_calls"]:
            fields = ("paper_a_id", "paper_b_id") if call["name"] == "paper_compare" else ("doc_id",) if call["name"] in {"paper_metadata", "paper_summary"} else ()
            if fields and any(call["args"].get(field) not in identifiers for field in fields):
                invalid.append({"id": row["id"], **call})
        if row["task_complete"] and len(results) == 1:
            event, question = results[0], QUESTIONS[row["id"]]["question"]
            evidence = event.get("result") or {}
            if (event["name"] == "knowledge_base_search" and event["status"] == "success"
                    and event["args"].get("question") == question and evidence.get("status") == "answered"
                    and evidence.get("generation_mode") == "grounded" and evidence.get("citations") and evidence.get("answer")):
                eligible.append(row["id"])
                if row["answer"] == evidence["answer"]:
                    retained.append(row["id"])
    return {"eligible_single_rag_questions": eligible, "exact_source_answer_retained": retained,
            "retention_rate": len(retained) / len(eligible) if eligible else None,
            "invalid_paper_id_calls": invalid, "invalid_paper_id_call_count": len(invalid),
            "failed_tool_calls": failed, "failed_tool_call_count": len(failed),
            "repeated_calls_questions": [r["id"] for r in rows if r["stop_reason"] == "repeated_calls"],
            "wrong_tool_path_questions": [r["id"] for r in rows if not r["tool_selection_correct"]]}


QUESTIONS = {q["id"]: q for q in read(ROOT / "reports/评测集.json")}


def main():
    baseline = read(BASE / "Agent两组结果_20261003.json")
    before = [r for r in baseline["rows"] if r["profile"] == "default"]
    retrieval = [r for r in read(BASE / "检索五组结果_20261003.json")["rows"] if r["profile"] == "rerank20"]
    manifest = read(ROOT / "reports/5_5_1 评测集构建/论文清单.json")
    identifiers = {p["doc_id"] for p in manifest["papers"]}
    corpus = read(ROOT / "data/raw/evaluation_vision_transformers/corpus.json")["corpus"]
    hashes = {p["id"]: p["doc_id"] for p in manifest["papers"]}
    b = {r["id"]: r for r in before}
    r = {r["id"]: r for r in retrieval}
    # 下面是依据实际原文的开发者定性复核，不是独立评阅人的0～4分。
    definitions = [
        ("BC55-01", "F005", ["生成错误", "Agent决策错误"],
         "金标准只列t2t_vit物理第2页；默认Top-1的第5页已明确T2T模块与T2T-ViT骨干，是标注页漏命中但语义有证据的例子。实际Agent把论文问题当一般概念直接回答，漏掉T2T模块与深窄骨干。",
         "规则路由的一般概念匹配优先于论文意图，绕过实际原文。固定证据页不是等价事实的穷尽标注，页级指标产生假阴性；不能把未命中标注页直接当语义检索失败。",
         "先确认论文意图再允许一般概念直答；后续由独立评阅人补充等价证据位置。本轮不修改冻结金标准、不改变检索与通用路由。"),
        ("BC55-02", "F002", ["生成错误"],
         "DeiT摘要明确说明仅用ImageNet、无外部数据；默认Top-5第3项为第1页图注，第4项为第2页贡献段，均明确no external data，工具仍声称没有相关文档并否定该训练设定。",
         "检索已提供明确限定条件，生成没有正确利用证据，并把有文档误说成空知识库；合法编号校验不能发现语义误拒答。",
         "区分空库/证据不足状态，针对问题限定条件核对证据；用原文相同输入验证生成，不靠降低相似阈值或人工改分。此轮保留为未解决项。"),
        ("BC55-03", "R001", ["生成错误", "引用错误"],
         "答案把缺少CNN归纳偏置时ViT在小数据上不易泛化，改成这些偏置使CNN表现不佳；引用编号来自ViT第2页，内容因果与主体错位。",
         "第2页开头是跨页续句，所引片段缺少上一页主体，可能促成主语误解；实际答案已发生主语与因果错位，记录不足以单独证明唯一成因。引用文件/页码存在不证明陈述受支持。",
         "保留句子前文或相邻块、检查主语/比较对象；引用语义仍需人工核验。此轮引用直通只避免二次丢失，不能校正RAG内部错误。"),
        ("BC55-04", "F001", ["引用错误"],
         "RAG给出vit.pdf物理第4与第6页的完整引用，Observation最终只保留页数，丢失文件名与正文标记。",
         "Agent重复生成最终答案，未强制保留RAG已格式化的来源。",
         "本轮实施：单次工具完整回答原问题且grounded、answered、有有效引用、Observation确认完成时，直接保留该RAG答案；多步骤/资料不足不适用。"),
        ("BC55-05", "R015", ["检索失败", "生成错误", "引用错误"],
         "问题限定只有ImageNet，应比较DeiT/T2T等起点；默认结果覆盖ViT但漏掉其他标注论文。模型将大规模预训练最佳结果解释为ImageNet-only，并把第6页不同模型规模表现相似说成ImageNet-21k与JFT预训练相似。",
         "跨论文覆盖不足；生成没有核对训练数据条件，将跨块指代their performances错误解释为数据集比较。来源编号合法，语义限定不支持该陈述。",
         "多论文问题按目标文献分别检索并汇总；保留性能表的训练条件、比较对象；逐条核对数字与限定条件。此轮不宣称语义引用已修复。"),
        ("BC55-06", "C006", ["Agent决策错误"],
         "paper_compare接收DETR、Deformable DETR名称而非SHA-256，两次输入失败后第三轮重复签名终止。",
         "Prompt仅靠模型自觉遵守ID契约，计划缺少取真实列表的前置步骤。",
         "本轮实施：必填论文ID工具缺少足够已知ID时先paper_list；仅从成功列表将唯一文件名或干名转为真实SHA；歧义/未知名称仍拒绝猜测。"),
        ("BC55-07", "S009", ["Agent决策错误"],
         "已执行paper_list，仍把vit.pdf、dino.pdf等文件名传入paper_compare，7轮后重复终止；三论文归纳题还被当成两论文对比。",
         "取ID与传ID之间没有校验桥接；名称解析失败与任务类型识别错误叠加。",
         "本轮修复唯一文件别名到真实ID的桥接；三论文归纳工具选择与多阶段任务完成判断继续作为剩余问题。"),
        ("BC55-08", "S013", ["Agent决策错误"],
         "真实Thought返回parallel_tools=[knowledge_base_search, knowledge_base_search]，两项名称重复而被拒绝；并未超过数量上限，没有执行工具。",
         "当前计划按工具名称去重，不能表示同一检索工具的两个独立参数调用；模型把按论文分别检索规划为重复名称，校验直接拒绝。",
         "后续以工具名+参数区分独立调用，在不超过2个线程的前提下允许同工具不同参数，或分轮处理多论文；重复相同参数仍终止。此轮未修改。"),
        ("BC55-09", "S014", ["生成错误", "Agent决策错误"],
         "Group是将论文按任务分组的英文动词，模型却生成不存在于本库的group_vit ID，失败后使用关键词工具并输出同义反复的任务名。",
         "英语指令被误识别为论文实体；失败替代工具虽成功，但并未完成原始论文归纳，完成标志发生误报。",
         "核对文献列表与原问题实体，论文归纳需以原文事实作答；独立评阅正确性/完整性。此次唯一别名解析不会把未知group_vit猜成ViT。"),
        ("BC55-10", "C013", ["检索失败", "生成错误"],
         "DeiT与CaiT图像级预测机制对比应包含蒸馏token和class-attention；Top-5三块来自PVT，DeiT块谈性能，CaiT块谈LayerScale与结论，没有目标机制。最终答案遗漏关键机制，还加入无关PVT。",
         "多论文目标机制未召回；泛化的image-level prediction关键词使PVT分类段排名靠前，未在固定Top-5为两篇目标论文保留证据；模型用邻近话题替代缺失机制。候选召回和重排的具体损失位置尚待导出完整候选进一步诊断。",
         "按用户提及论文限定检索，分别召回机制章节后合并；评测多论文证据覆盖，不只看任意一页Hit。本轮检索未改，作为后续优先项。"),
    ]
    cases = []
    for code, ident, categories, observed, cause, proposal in definitions:
        question = QUESTIONS[ident]
        sources = []
        for evidence in question["evidence"]:
            chunks = [c for c in corpus if c["metadata"]["doc_id"] == hashes[evidence["paper_id"]]
                      and c["metadata"]["page_number"] <= evidence["page_number"] <= c["metadata"].get("page_end", c["metadata"]["page_number"])]
            sources.append({"gold": evidence, "chunks": chunks})
        cases.append({"case_id": code, "question_id": ident, "categories": categories,
                      "verification": "原文及实测逐例开发复核，非独立人工评分", "observed": observed,
                      "root_cause": cause, "improvement": proposal, "question": question,
                      "baseline_agent": b[ident], "baseline_retrieval": r[ident], "original_evidence": sources})
    report = {"created_at": datetime.now().astimezone().isoformat(), "baseline_questions": 60,
              "classification_nonexclusive": True, "human_scoring": "pending_independent_review",
              "retrieval_no_gold_page_hit": [x["id"] for x in retrieval if not x["hit_at_5"]],
              "retrieval_incomplete_gold_paper_coverage": [x["id"] for x in retrieval if not x["all_papers_hit_at_5"]],
              "structural_before": structural(before, identifiers), "confirmed_representative_cases": cases}
    evaluation.write_json(HERE / "Bad_Case清单.json", report)
    after_path = HERE / "Agent优化后60题_20261003.json"
    if after_path.exists() and read(after_path)["status"] == "completed":
        after = read(after_path)["rows"]
        assert len(after) == 60 and {x["id"] for x in after} == set(QUESTIONS)
        comparison = {"created_at": datetime.now().astimezone().isoformat(), "questions": 60,
                      "before": {"metrics": evaluation.summarize_agent(before), "structural": structural(before, identifiers)},
                      "after": {"metrics": evaluation.summarize_agent(after), "structural": structural(after, identifiers)},
                      "paired": [{"id": x["id"], "question": QUESTIONS[x["id"]], "before": b[x["id"]], "after": x} for x in after],
                      "scope": "同集开发迭代；不将完成率/引用保留率当作人工答案质量或引用准确性"}
        evaluation.write_json(HERE / "优化前后对比数据.json", comparison)
    print(f"已分类10个代表案例；无标注页命中{len(report['retrieval_no_gold_page_hit'])}/60，论文覆盖不全{len(report['retrieval_incomplete_gold_paper_coverage'])}/60。")


if __name__ == "__main__":
    main()
