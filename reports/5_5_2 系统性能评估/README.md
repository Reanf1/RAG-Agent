# 5.5.2 系统性能评估

成绩／环境／图表只在[系统评测报告](../系统评测报告.md)维护，指标定义与评阅规则见[5.5.2 QA](../../docs/QA/5.5.2%20系统性能评估.md)。本目录保存复现输入与原始证据，不重复成绩表。

## 数据与核验

[评测方案](评测方案.json)、[环境](运行环境_20261003.json)、[300条检索](检索五组结果_20261003.json)、[120条Agent基线](Agent两组结果_20261003.json)、[626次实际响应](Agent两组结果_20261003.calls.jsonl)、[公式性能表](系统性能评测数据.xlsx)、[数值复核](评测数值复核_20261003.json)、[文件核验](交付文件核验_20261003.json)。优化60条见[5.5.3](../5_5_3%20Bad%20Case分析与优化/README.md)。

正式[180条人评表](答案质量人工评分表_180条.xlsx)含default／no_rules／optimized各60条，旧[120条表](答案质量人工评分表.xlsx)留为历史。[答案来源](人工评分180条来源.json)与[0/180待填核验](人工评分180条待填核验_20261004.json)保留，质量均值未知；0有效、空白待评，不以助手规则评分替人评。

## 复现

先准备本地12篇固定原文、M3E、BGE、Qwen与Ollama服务。使用新工作目录和新结果文件，保留首次失败及已有原始结果。

```bash
# 模块一准备与证据核验（新文件名）。
.venv/bin/python "reports/5_5_1 评测集构建/verify_dataset.py" --output "reports/5_5_1 评测集构建/核验_新时间.json"

# 构建隔离索引，跑五组检索；root必须不存在。
.venv/bin/python "reports/5_5_2 系统性能评估/evaluate_system.py" --stage retrieval --root /private/tmp/rag-eval-新时间 --output "reports/5_5_2 系统性能评估/检索_新时间.json"

# 复用同一隔离索引和完全相同的输入/源码，跑同60题的两组Agent。
.venv/bin/python "reports/5_5_2 系统性能评估/evaluate_system.py" --stage agent --root /private/tmp/rag-eval-新时间 --output "reports/5_5_2 系统性能评估/Agent_新时间.json"

# 只验证指标边界，不调用模型，不算作真实系统评测样本。
.venv/bin/python -m unittest discover -s "tests/5_5_2 系统性能评估" -p 'test_*.py' -v
```

`--limit`只用于先行验证，正式数据为60题。评测期间不要同时运行其它模型或回归任务，以免干扰延迟；原始报告记录实际运行环境。工作目录保存原文复制件、Chroma与隔离业务日志，后续可以定位真实工具结果。

`plot_system.py`读取完成的检索/Agent JSON生成三幅PNG，不重新调用模型。`build_workbooks.mjs`使用Codex内置Artifact Tool生成可复算性能表和人工评阅表，`summarize_human_scores.mjs`只汇总评阅人填回的xlsx；报告脚本不是业务依赖，不修改requirements.txt。

人工表回传后，用内置Node及其`node_modules`依赖执行`summarize_human_scores.mjs 人工评分表.xlsx Agent结果.json 新人工汇总.json`。未填写的记录保持待评分；部分填写或越界分数会报出题号供修正。无需重新调用模型，也不覆盖原始空白表。内置运行时不在仓库依赖中；普通环境可按表中相同规则人工汇总。

## 失败与后续

首次[先行检索](先行检索验证_20261003.log)仅中文却错误要求英文组，修正按实际分组；[Agent预热](先行Agent验证_20261003.log)沙箱Metal失败，获准本地环境后修正，先行不进正式成绩。字体缺失图片保留在首次图表字体缺失目录，修正用Arial Unicode，[字体日志](图表字体修正_20261003.log)可核对。

真正固定RAG对照[144条Mac记录](Agent与固定RAG对照_20261004.json)与[串并行3/12暂停](独立工具串行并行对照_20261004.json)另存，不用关闭规则组替代固定RAG，不将未完成并行结果汇总成提速结论。当前暂停真实模型／Windows性能测试。

[返回报告索引](../README.md)
