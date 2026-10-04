"""汇总已经完成的三个研究问题，保留拒答、调度失败及资源限制。"""
import json
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'reports/5_5_2 系统性能评估'


def main():
    routing=json.loads((FOLDER/'Agent与固定RAG对照_20261004.json').read_text())
    parallel=json.loads((FOLDER/'独立工具串行并行对照_20261004.json').read_text())
    chunk=json.loads((ROOT/'reports/5_1_2 文本分块策略/五组分块检索对比_20261004.json').read_text())
    assert routing['status']==parallel['status']==chunk['status']=='completed'
    a,f=routing['profiles']['agent'],routing['profiles']['fixed_rag']
    token_delta=(a['tokens_mean_known']/f['tokens_mean_known']-1)*100
    lines=['# 三个研究问题对照报告（2026年10月4日）','','## 研究问题一：分块、返回K和模型选择','','[五组分块完整实验](../5_1_2%20文本分块策略/五组分块检索对比实验报告_20261004.md)固定12篇/60题与M3E/RRF20/BGE，300查询、900组K指标独立复算。递归512/64的Hit@5 66.67%、Recall@5 42.97%最高，固定512 MRR@5 0.3708更高；返回10块递归Hit@10 80%。保留递归与Top-5，不认定所有指标全面最优。M3E/BGE Embedding中英AI论文对照已在5.1.3完成，用户指定M3E；本轮不更换或重跑选型。','','## 研究问题二：Agent与每次固定RAG','','60题原论文集加12题（概念、计算、时间各4），72题成对交错，共144请求。两模式同一真实索引、模型及问题；独立空历史，固定种子20261004。固定RAG每题实际调用knowledge_base_search，不使用Agent，低相关时按现有产品返回用户确认，缺少有效引用时标记资料不足。没有把“关闭规则但仍运行Agent”冒充固定RAG。','','| 模式 | 请求数 | 平均实际Token | 未知用量请求 | 平均完整耗时 / 秒 |','|---|---:|---:|---:|---:|']
    for label,p in [('Agent统一决策',a),('每次固定RAG',f)]:
        lines.append(f"| {label} | {p['requests']} | {p['tokens_mean_known']:.2f} | {p['unknown_requests']} | {p['seconds_mean']:.3f} |")
    lines+=['',f"全部请求平均Token：Agent相对固定RAG{'增加' if token_delta>=0 else '减少'}{abs(token_delta):.2f}%。不把概念/计算/时间不检索等同于必然节省总Token；Agent仍要执行Action/Observation。固定RAG的低相关确认可能0 Token，所有拒答与失败留在分母，不强迫它产生无依据答案。",'','| 类型 | 每模式题数 | Agent实际Token均值 | 固定RAG实际Token均值 | Agent路径准确率 |','|---|---:|---:|---:|---:|']
    for category in a['by_category']:
        x,y=a['by_category'][category],f['by_category'][category]
        lines.append(f"| {category} | {x['requests']} | {x['tokens_mean_known']:.2f} | {y['tokens_mean_known']:.2f} | {x['tool_path_accuracy']:.2%} |")
    accuracies={name:part['tool_path_accuracy'] for name,part in a['by_category'].items()}
    highest,lowest=max(accuracies.values()),min(accuracies.values())
    high_names='、'.join(name for name,value in accuracies.items() if value==highest)
    low_names='、'.join(name for name,value in accuracies.items() if value==lowest)
    complete={p:sum(r['task_complete'] for r in routing['rows'] if r['profile']==p) for p in ['agent','fixed_rag']}
    lines+=['','用户确认的180条人评范围是此前基线120条及优化后60条；本次144条路由与12条并行请求不计入这180条，本报告只核验它们的过程、工具、实际用量和延迟，不声称已经人工评价语义质量。']
    lines+=['',f"本集路径代理准确率最高类型为{high_names}（{highest:.2%}），最低为{low_names}（{lowest:.2%}）。类型样本量不同，补充类型各仅4题，不能据此推断其它问题的稳定表现。",'',f"Agent路径代理准确率{a['tool_path_accuracy']:.2%}，平均{a['iterations_mean']:.4f}轮。代理按预先定义的工具及论文ID覆盖判定，不能当答案正确率。固定RAG本来不负责Agent式路由，不能把它按Agent工具规则计算的路径分数用于宣称准确率优劣。模型自报task_complete仅作过程记录：Agent {complete['agent']}/72、固定RAG {complete['fixed_rag']}/72；180条质量评分尚未完成。",'','![Agent与固定RAG](Agent与固定RAG对照_20261004.png)','','## 研究问题三：常用工具组合与真实并行延迟','','路由实验72个Agent请求的工具组合按每请求去重汇总，重复调用不扩大组合次数：','','| 工具组合 | 请求数 |','|---|---:|']
    for combo,count in sorted(routing['tool_combinations'].items(),key=lambda x:-x[1]):lines.append(f'| {combo} | {count} |')
    lines+=['','另外真实运行两种独立任务：current_time＋给定文本keyword_extract；同一knowledge_base_search工具分别处理ViT、DeiT两套已知输入。每模式各3次，共12个完整Agent请求，交错串行/并行顺序。只改变execute_calls的实际调度，模型规划、工具、参数、Observation仍走业务代码，没有mock工具结果或人工拼接延迟。','','| 组合 | 串行均值 / 秒 | 并行均值 / 秒 | 并行相对变化 | 两项独立成功的串行/并行次数 | 自报完成串行/并行次数 |','|---|---:|---:|---:|---|---|']
    audit=json.loads((FOLDER/'独立工具串行并行对照_20261004_独立复核.json').read_text())
    failed={(r['id'],r['profile'],r['repeat']) for r in audit['schedule_failures']}
    condensed=[]
    for category in ['time_keywords','two_knowledge']:
        parts={p:[r for r in parallel['rows'] if r['category']==category and r['profile']==p] for p in ['serial','parallel']}
        secs={p:mean(r['seconds'] for r in part) for p,part in parts.items()}
        delta=(secs['parallel']/secs['serial']-1)*100
        success={p:sum((r['id'],r['profile'],r['repeat']) not in failed for r in part) for p,part in parts.items()}
        completed={p:sum(r['task_complete'] for r in part) for p,part in parts.items()}
        lines.append(f"| {category} | {secs['serial']:.3f} | {secs['parallel']:.3f} | {'增加' if delta>=0 else '减少'}{abs(delta):.2f}% | {success['serial']}/3，{success['parallel']}/3 | {completed['serial']}/3，{completed['parallel']}/3 |")
        condensed.append(f"{category}串行{secs['serial']:.3f}秒、并行{secs['parallel']:.3f}秒，两项独立成功次数分别{success['serial']}/3和{success['parallel']}/3。")
    lines+=['','工具调度成功需实际出现两个不同输入、两个结果、成功状态及目标execution_mode；模型最终自报完成另列，不混为一谈。Ollama只允许1个生成并发，两个RAG生成可能排队；快工具自身耗时短，完整LLM规划和观察会掩盖调度收益。样本只有3对/组合，不作显著性或普遍加速结论。',f"独立调度核验失败记录{len(failed)}条，完整保留，不能把失败快速返回计为并行提速。",'','![两模式实际调度](独立工具串行并行对照_20261004.png)','','## 证据及复现边界','','逐题实际HTTP响应和prompt_eval_count/eval_count保存在各`.calls.jsonl`；缺少计数保持未知。汇总不扣除失败，不估算Token。[路由原始结果](Agent与固定RAG对照_20261004.json)、[路由独立复核](Agent与固定RAG对照_20261004_独立复核.json)、[并行原始结果](独立工具串行并行对照_20261004.json)、[并行独立复核](独立工具串行并行对照_20261004_独立复核.json)。输入、模型版本、种子和启动时源码快照SHA-256均留存，后续页面文案更新不改写历史哈希。','','本机M4/16GiB，Qwen7b Metal、M3E/BGE CPU4线程；路由与分块及回归曾共享资源，6GB Colima在15:17停止，分块续跑结束后只保留路由推理。时间是实际工程观测，不能当严格隔离、重复采样的硬件基准。并行实验在路由完成后独立启动，仍保留本机环境噪声与冷/热状态限制。','','```bash','.venv/bin/python "reports/5_5_2 系统性能评估/evaluate_research.py" --stage routing --root /tmp/rag-route-new --output /tmp/rag-route-new.json','.venv/bin/python "reports/5_5_2 系统性能评估/evaluate_research.py" --stage parallel --root /tmp/rag-parallel-new --output /tmp/rag-parallel-new.json','```','','每次用新目录，避免覆盖用户知识库与既有结果；需保留原冻结12篇索引和本地模型。脚本中冻结索引路径在当前开发机器，换机器先准备相同论文、相同配置及块ID，不能复制不同索引冒充同实验。']
    lines+=['','换机器复现可先用已有检索评测脚本准备相同12篇的冻结目录，再显式传入新加入的--frozen-root；默认路径保留本轮历史环境。该参数只指定输入目录，不改变模型、问题、调度或指标。','','```bash','.venv/bin/python "reports/5_5_2 系统性能评估/evaluate_system.py" --stage retrieval --root /tmp/rag-eval --output /tmp/rag-eval-retrieval.json','.venv/bin/python "reports/5_5_2 系统性能评估/evaluate_research.py" --stage routing --frozen-root /tmp/rag-eval --root /tmp/rag-route-new --output /tmp/rag-route-new.json','.venv/bin/python "reports/5_5_2 系统性能评估/evaluate_research.py" --stage parallel --frozen-root /tmp/rag-eval --root /tmp/rag-parallel-new --output /tmp/rag-parallel-new.json','```']
    (FOLDER/'研究问题对照报告_20261004.md').write_text('\n'.join(lines)+'\n')
    word='''#### 4.5.1 分块和返回K对照\n\n固定12篇/60题与相同模型，五组300查询、K=3/5/10共900组指标复核。递归Hit@5 66.67%、Recall@5 42.97%最高，固定512 MRR@5 0.3708更高。默认保留递归512/64与返回5块。\n\n![五组分块及K](../reports/5_1_2%20文本分块策略/五组分块检索对比_20261004.png)\n\n#### 4.5.2 Agent对固定RAG\n\n'''
    word+=f"72题成对交错共144请求：Agent平均{a['tokens_mean_known']:.2f}实际Token、{a['seconds_mean']:.3f}秒；固定RAG平均{f['tokens_mean_known']:.2f} Token、{f['seconds_mean']:.3f}秒。Agent总Token相对{'增加' if token_delta>=0 else '减少'}{abs(token_delta):.2f}%，不据宣传假定80%节省。固定RAG拒答/低相关确认可能0 Token，保留失败分母，答案质量仍待人工评分。Agent路径代理准确率{a['tool_path_accuracy']:.2%}，不当正确率。\n\n![Agent对固定RAG](../reports/5_5_2%20系统性能评估/Agent与固定RAG对照_20261004.png)\n\n#### 4.5.3 两组合串行与并行\n\n"
    word+=''.join(condensed)+f"实际12个完整请求；调度失败{len(failed)}条保留。模型单并发可能排队，仅3对/组合，不宣称普遍加速或显著性。工具组合统计、原始响应和独立复算见研究问题对照报告。\n\n![串行与并行](../reports/5_5_2%20系统性能评估/独立工具串行并行对照_20261004.png)"
    p=ROOT/'docs/课程报告.md';s=p.read_text().replace('<!-- DELIVERY_RESEARCH_RESULTS -->',word)
    s=s.replace('| 三研究问题对照 | — | — | 补充分块 K 值、固定 RAG 与并行实验，最终数值见第四章 |','| 三研究问题对照 | 是 | — | 分块 K 值、固定 RAG、并行均真实运行；失败与限制见第四章 |')
    p.write_text(s)
    print('已按已完成的实际结果生成研究报告和课程报告小节')


if __name__=='__main__': main()
