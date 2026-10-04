# 5.5.4 项目交付

**部署与测试已暂停（2026年10月4日）**：按用户要求停止Mac实验，待Windows部署后继续；Mac耗时仅作开发记录，不作为最终性能依据。[暂停交接说明](暂停与Windows复测说明_20261004.md)。

本轮依据两份原始课程Word核对缺项，补齐五组分块、真正固定RAG对照、多工具完整并行对照、正文流式、PDF原文页、容器部署及课程报告/演示。

## 目前可核验的资料

- [交付补齐QA](../../docs/QA/项目交付补齐.md)。
- [全量668项通过](全量668项通过_20261004.log)，首次缩进失败日志保留；[模块四183项专项](模块四补齐专项_20261004.json)包含在全量之内。
- [五组分块/K真实实验](../5_1_2%20文本分块策略/五组分块检索对比实验报告_20261004.md)。
- [Agent流式与原文页实际证据](../5_4_3%20前端与会话管理/交付补齐_20261004/README.md)。
- [180条评分表](../5_5_2%20系统性能评估/答案质量人工评分表_180条.xlsx)；按用户选择包含基线120与优化60，0/180已评，分数未知。
- [课程报告Markdown](../../docs/课程报告.md)；正式Word和PPT待最终实验与容器验证后生成，初稿不作正式交付。

## 复现脚本

独立实验目录避免覆盖用户索引。评测需准备config.yaml约定的本地M3E、BGE、Qwen及词表；实际用量读Ollama响应。验证脚本不修改金标准。

```bash
.venv/bin/python "reports/5_5_4 项目交付/verify_experiments.py" chunking
.venv/bin/python "reports/5_5_4 项目交付/verify_experiments.py" routing
.venv/bin/python "reports/5_5_4 项目交付/verify_experiments.py" parallel
```

`plot_delivery.py`按同一stage生成研究图表；尚未完成的JSON会明确拒绝绘图。`verify_container.py`只能在独立验收容器运行，做真实PDF导入、索引、重排、Agent问答、原文页、会话持久化和外网TCP阻断检查。

Word用`build_course_report.py`从原模板副本生成，按实际渲染页号生成可点击目录；成员贡献和教师评分保留待填。`build_demo.cjs`以真实结果和截图生成14页演示，最终模式拒绝未完成实验或未通过部署记录。作者工具使用Codex打包文档/演示运行时，不将额外制图依赖塞入应用requirements。
