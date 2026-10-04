# 按课程小节组织的自动化测试

截至 2026-10-03，既有 **633 个测试**按 **16 个课程小节、42 个测试文件**拆分。保留原来的测试类、样例和断言；只调整目录层级、共用样例导入和日志隔离位置。此前按小节重构时业务源码没有修改。

5.5.2新增10项指标边界测试，单独运行通过；该阶段共643个用例、17个小节、43个测试文件。既有633项全量回归记录仍作为原日期的历史结果，不把新增指标测试当作系统答案质量成绩。

5.5.3再新增11项Bad Case交接回归，修复前4项失败，修复后通过；该阶段共**654个用例、18个小节、44个测试文件**。2026-10-04完整654项通过（43.662秒，无失败/错误/跳过），见[本轮日志](../reports/5_5_3%20Bad%20Case分析与优化/完整654项回归测试.log)。测试HTTP为mock，实际模型60题及残留质量问题另见Bad Case报告。

目录采用“编号 空格 中文小节名”，例如 `5_1_1 文档加载与批量导入`。测试包的 `__init__.py` 用于标准 `unittest` 递归发现，含空格的路径须加引号。

本轮交付增加正文流式、原文物理页、历史来源和同名独立并行共14项回归；现有**668个用例、18个小节、46个测试文件**，2026-10-04全量通过（59.730秒，无跳过）。[本轮通过日志](../reports/5_5_4%20项目交付/全量668项通过_20261004.log)，首次文案缩进失败记录保留。功能测试与实际模型评测分开报告。

本次科研对话改版移除独立单轮RAG的UI测试，补入消息框、累计统计与异常历史测试；5.4.3当前35项通过，连同指标/文档流程及模块二保留能力，共172项不同测试通过。上述668项为此前完整回归记录，本轮未宣称全量重新运行，见[本轮验证](../reports/5_4_3%20前端与会话管理/科研对话改版_20261004/验证说明.md)。下表其余数量为历史归档口径。

## 小节与测试文件

| 小节 | 用例数 | 测试文件 |
| --- | ---: | --- |
| 5_1_1 文档加载与批量导入 | 39 | [test_pdf_loader.py](5_1_1%20文档加载与批量导入/test_pdf_loader.py)、[test_docx_loader.py](5_1_1%20文档加载与批量导入/test_docx_loader.py)、[test_text_loader.py](5_1_1%20文档加载与批量导入/test_text_loader.py)、[test_batch_import.py](5_1_1%20文档加载与批量导入/test_batch_import.py) |
| 5_1_2 文本分块策略 | 38 | [test_chunking.py](5_1_2%20文本分块策略/test_chunking.py)、[test_academic_pdf.py](5_1_2%20文本分块策略/test_academic_pdf.py) |
| 5_1_3 向量化与存储 | 30 | [test_vector_store.py](5_1_3%20向量化与存储/test_vector_store.py)、[test_embeddings.py](5_1_3%20向量化与存储/test_embeddings.py)、[test_batch_index.py](5_1_3%20向量化与存储/test_batch_index.py) |
| 5_1_4 混合检索与重排序 | 42 | [test_bm25.py](5_1_4%20混合检索与重排序/test_bm25.py)、[test_hybrid_retriever.py](5_1_4%20混合检索与重排序/test_hybrid_retriever.py)、[test_reranker.py](5_1_4%20混合检索与重排序/test_reranker.py)、[test_retrieval_evaluation.py](5_1_4%20混合检索与重排序/test_retrieval_evaluation.py) |
| 5_2_1 Prompt工程与生成策略 | 57 | [test_prompt.py](5_2_1%20Prompt工程与生成策略/test_prompt.py)、[test_context.py](5_2_1%20Prompt工程与生成策略/test_context.py)、[test_citations.py](5_2_1%20Prompt工程与生成策略/test_citations.py)、[test_generation.py](5_2_1%20Prompt工程与生成策略/test_generation.py) |
| 5_2_2 流式输出与引用 | 17 | [test_streaming.py](5_2_2%20流式输出与引用/test_streaming.py) |
| 5_2_3 缓存与降级策略 | 20 | [test_degradation.py](5_2_3%20缓存与降级策略/test_degradation.py)、[test_cache.py](5_2_3%20缓存与降级策略/test_cache.py) |
| 5_2_4 日志与可观测性 | 8 | [test_rag_logging.py](5_2_4%20日志与可观测性/test_rag_logging.py) |
| 5_3_1 Agent核心循环 | 48 | [test_thought.py](5_3_1%20Agent核心循环/test_thought.py)、[test_action.py](5_3_1%20Agent核心循环/test_action.py)、[test_observation.py](5_3_1%20Agent核心循环/test_observation.py)、[test_system_prompt.py](5_3_1%20Agent核心循环/test_system_prompt.py) |
| 5_3_2 工具集开发 | 87 | [test_research_tools.py](5_3_2%20工具集开发/test_research_tools.py)、[test_comparison_keywords.py](5_3_2%20工具集开发/test_comparison_keywords.py)、[test_summary_time_search.py](5_3_2%20工具集开发/test_summary_time_search.py)、[test_calculator_paper_list.py](5_3_2%20工具集开发/test_calculator_paper_list.py) |
| 5_3_3 Agent决策优化 | 46 | [test_routing_parallel.py](5_3_3%20Agent决策优化/test_routing_parallel.py)、[test_error_recovery.py](5_3_3%20Agent决策优化/test_error_recovery.py) |
| 5_3_4 多轮对话记忆管理 | 55 | [test_session_isolation.py](5_3_4%20多轮对话记忆管理/test_session_isolation.py)、[test_history_window.py](5_3_4%20多轮对话记忆管理/test_history_window.py)、[test_conversation_summary.py](5_3_4%20多轮对话记忆管理/test_conversation_summary.py) |
| 5_4_1 RAG与Agent深度融合 | 28 | [test_source_routing.py](5_4_1%20RAG与Agent深度融合/test_source_routing.py)、[test_memory_context.py](5_4_1%20RAG与Agent深度融合/test_memory_context.py) |
| 5_4_2 可观测性与健康检查 | 56 | [test_health_check.py](5_4_2%20可观测性与健康检查/test_health_check.py)、[test_agent_metrics.py](5_4_2%20可观测性与健康检查/test_agent_metrics.py) |
| 5_4_3 前端与会话管理 | 35 | [test_document_management.py](5_4_3%20前端与会话管理/test_document_management.py)、[test_chat_streaming.py](5_4_3%20前端与会话管理/test_chat_streaming.py)、[test_conversation_history.py](5_4_3%20前端与会话管理/test_conversation_history.py)、[test_agent_streaming.py](5_4_3%20前端与会话管理/test_agent_streaming.py)、[test_citation_page.py](5_4_3%20前端与会话管理/test_citation_page.py) |
| 5_4_4 端到端联调与测试 | 28 | [test_document_flow.py](5_4_4%20端到端联调与测试/test_document_flow.py) |
| 5_5_2 系统性能评估 | 10 | [test_system_evaluation.py](5_5_2%20系统性能评估/test_system_evaluation.py) |
| 5_5_3 Bad Case分析与优化 | 11 | [test_bad_case_regression.py](5_5_3%20Bad%20Case分析与优化/test_bad_case_regression.py) |

文档上传、索引构建、删除恢复和上传边缘场景的 AppTest 用例归入 5.4.4；组件、聊天和历史会话操作归入 5.4.3。会话并发隔离用例与会话存储测试一起保留在 5.3.4。模块四专项入口仍按实际职责选择 170 个用例，不按目录重复统计。

## 运行方式

在项目根目录执行全部测试：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

只运行一个课程小节（此例 39 项）：

```bash
.venv/bin/python -m unittest discover -s 'tests/5_1_1 文档加载与批量导入' -p 'test_*.py' -v
```

直接运行一个文件，或指定该文件中的测试类：

```bash
.venv/bin/python 'tests/5_1_1 文档加载与批量导入/test_pdf_loader.py' -v
.venv/bin/python 'tests/5_3_1 Agent核心循环/test_action.py' TestAction -v
```

从 `src/`、`/private/tmp` 等项目外目录执行时使用绝对路径。每个测试文件按自身位置加入项目根目录，避免 `ModuleNotFoundError: No module named 'src'`：

```bash
/Users/rean/github/RAG+Agent/.venv/bin/python '/Users/rean/github/RAG+Agent/tests/5_1_1 文档加载与批量导入/test_pdf_loader.py' -v
```

模块四专项或带 JSON 记录的全量回归继续使用已有入口，输出路径须不存在：

```bash
.venv/bin/python reports/模块完整性验证/verify_completeness_tests.py --scope module4 --output reports/模块完整性验证/模块四专项_新时间.json
.venv/bin/python reports/模块完整性验证/verify_completeness_tests.py --scope all --output reports/模块完整性验证/全量回归_新时间.json
```

## 样例和验证边界

- PDF/Word/文本样例由测试明确生成，验证实际文件加载和来源位置；Chroma、SQLite、线程与 Streamlit AppTest 使用临时数据实际执行。
- [helpers.py](helpers.py) 只放跨小节复用的二维模拟向量、NDJSON 流式响应及 Agent 临时日志上下文。模拟向量不表示 M3E 的真实效果，构造 HTTP 响应不表示真实模型答案质量或服务可用。
- 每个 Agent 测试模块在开始和结束时进入/释放临时日志上下文；既有 `patch.stopall()` 不会撤销该日志隔离。会话 Token 窗口仍使用本地官方 Qwen 词表。
- 需要项目 `.venv` 中的依赖和已准备的本地词表；自动化测试不启动 Ollama，也不重新执行模型性能实验。
- 真实模型、真实浏览器、并发和边缘场景验证脚本及原始失败记录继续位于 [reports](../reports/README.md)。633 项自动化通过不代表全部课程验收或答案引用质量通过。

## 本次重构记录

原 `test_retrieval.py`、`test_generation.py`、`test_agent.py` 已拆分，不再作为运行入口。旧 QA 中的数量、耗时和报告中的测试标识是当时记录，保持历史结论；文档中的可执行命令和文件链接已同步到当前小节。

重构后全量 633 项通过，失败、错误及跳过为 0，耗时 40.538 秒。16 个小节独立发现数量一致；从项目外目录运行 6 个代表文件及 39 项文档加载小节均通过。详见 [完整回归结果](../reports/模块完整性验证/测试按小节重构全量回归_20261003.json)、[运行日志](../reports/模块完整性验证/测试按小节重构全量回归_20261003.log)与[迁移及入口核验](../reports/模块完整性验证/测试按小节重构核验_20261003.json)。
