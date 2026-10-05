# 5.5.4 项目交付

当前状态统一见[5.5.4 QA](../../docs/QA/5.5.4%20项目交付.md)及[技术设计验收](../../docs/技术设计文档.md)。按用户要求暂停Windows、Mac模型性能与容器测试；历史原始资料及性能结果保留。

## 当前证据

- [文档合并与精简记录](文档整理记录_20261005.md)。

- [Mac可修复问题与提交](Mac可修复问题记录_20261004.md)，最新[688项自动通过](../5_2_3%20缓存与降级策略/候选确认接入_20261004/全量688项通过.log)，不是Windows现场复测。
- [Windows历史失败报告](../5_4_4%20端到端联调与测试/Windows浏览器验收_20261004/Windows浏览器验收报告.md)及[历史暂停交接](暂停与Windows复测说明_20261004.md)。交接中R01“待修复”为当时状态，源码已修复但真实对照未重跑。
- [合并分块报告](../5_1_2%20文本分块策略/分块策略对比实验报告.md)、[Agent正文／原文页真实证据](../5_4_3%20前端与会话管理/交付补齐_20261004/README.md)。
- [180条人工表](../5_5_2%20系统性能评估/答案质量人工评分表_180条.xlsx)：0/180待评；成员信息与真实贡献待本人提供。
- [课程报告Markdown](../../docs/课程报告.md)，Word／14页PPT有初稿，最终模式需完整实验／部署后生成，不能按初稿宣称已交付。

## 复算与构建

已有模型／论文的实验使用独立目录，以下脚本从归档记录复算；未完成状态会拒绝生成完整成绩图。

```bash
.venv/bin/python "reports/5_5_4 项目交付/verify_experiments.py" chunking
.venv/bin/python "reports/5_5_4 项目交付/verify_experiments.py" routing
.venv/bin/python "reports/5_5_4 项目交付/verify_experiments.py" parallel
```

`plot_delivery.py`按stage绘图。`verify_container.py`仅在独立验收容器实际检查导入／索引／重排／问答／会话重开和外网阻断，当前尚未完整运行通过。

`build_course_report.py`从原模板副本生成Word与实际目录页号，`build_demo.cjs`生成演示，`audit_delivery.py`核对原件、实验、部署及版式；最终构建拒绝未完成输入。作者工具依赖不写入业务requirements。旧668项／183项日志仍为历史阶段，最新回归在上方。

[返回总索引](../README.md)
