# 自动化测试

测试按“5_1_1 中文小节名”组织，目录含空格时命令路径加引号。当前707个用例、18个小节、47个测试文件；2026-10-05 Windows问题本地修复后全量通过，无失败／错误／跳过，见[完整日志](../reports/5_5_4%20项目交付/Windows问题修复_20261005/全量回归.log)及[逐项结果](../reports/5_5_4%20项目交付/Windows问题修复_20261005/全量回归.json)。

此前代码精简保持688项；本轮新增19项实际缺陷回归，并同步无参数文献列表的旧HTTP模拟样例。首次705项回归的旧样例失败保留，见[问题修复记录](../reports/5_5_4%20项目交付/Windows问题修复记录_20261005.md)。

## 小节与入口

| 小节 | 用例数 | 测试文件 |
| --- | ---: | --- |
| 5_1_1 文档加载与批量导入 | 41 | [test_pdf_loader.py](5_1_1%20文档加载与批量导入/test_pdf_loader.py)、[test_docx_loader.py](5_1_1%20文档加载与批量导入/test_docx_loader.py)、[test_text_loader.py](5_1_1%20文档加载与批量导入/test_text_loader.py)、[test_batch_import.py](5_1_1%20文档加载与批量导入/test_batch_import.py) |
| 5_1_2 文本分块策略 | 38 | [test_chunking.py](5_1_2%20文本分块策略/test_chunking.py)、[test_academic_pdf.py](5_1_2%20文本分块策略/test_academic_pdf.py) |
| 5_1_3 向量化与存储 | 30 | [test_vector_store.py](5_1_3%20向量化与存储/test_vector_store.py)、[test_embeddings.py](5_1_3%20向量化与存储/test_embeddings.py)、[test_batch_index.py](5_1_3%20向量化与存储/test_batch_index.py) |
| 5_1_4 混合检索与重排序 | 43 | [test_bm25.py](5_1_4%20混合检索与重排序/test_bm25.py)、[test_hybrid_retriever.py](5_1_4%20混合检索与重排序/test_hybrid_retriever.py)、[test_reranker.py](5_1_4%20混合检索与重排序/test_reranker.py)、[test_retrieval_evaluation.py](5_1_4%20混合检索与重排序/test_retrieval_evaluation.py) |
| 5_2_1 Prompt工程与生成策略 | 58 | [test_prompt.py](5_2_1%20Prompt工程与生成策略/test_prompt.py)、[test_context.py](5_2_1%20Prompt工程与生成策略/test_context.py)、[test_citations.py](5_2_1%20Prompt工程与生成策略/test_citations.py)、[test_generation.py](5_2_1%20Prompt工程与生成策略/test_generation.py) |
| 5_2_2 流式输出与引用 | 17 | [test_streaming.py](5_2_2%20流式输出与引用/test_streaming.py) |
| 5_2_3 缓存与降级策略 | 30 | [test_degradation.py](5_2_3%20缓存与降级策略/test_degradation.py)、[test_cache.py](5_2_3%20缓存与降级策略/test_cache.py)、[test_agent_cache.py](5_2_3%20缓存与降级策略/test_agent_cache.py) |
| 5_2_4 日志与可观测性 | 9 | [test_rag_logging.py](5_2_4%20日志与可观测性/test_rag_logging.py) |
| 5_3_1 Agent核心循环 | 56 | [test_thought.py](5_3_1%20Agent核心循环/test_thought.py)、[test_action.py](5_3_1%20Agent核心循环/test_action.py)、[test_observation.py](5_3_1%20Agent核心循环/test_observation.py)、[test_system_prompt.py](5_3_1%20Agent核心循环/test_system_prompt.py) |
| 5_3_2 工具集开发 | 92 | [test_research_tools.py](5_3_2%20工具集开发/test_research_tools.py)、[test_comparison_keywords.py](5_3_2%20工具集开发/test_comparison_keywords.py)、[test_summary_time_search.py](5_3_2%20工具集开发/test_summary_time_search.py)、[test_calculator_paper_list.py](5_3_2%20工具集开发/test_calculator_paper_list.py) |
| 5_3_3 Agent决策优化 | 47 | [test_routing_parallel.py](5_3_3%20Agent决策优化/test_routing_parallel.py)、[test_error_recovery.py](5_3_3%20Agent决策优化/test_error_recovery.py) |
| 5_3_4 多轮对话记忆管理 | 55 | [test_session_isolation.py](5_3_4%20多轮对话记忆管理/test_session_isolation.py)、[test_history_window.py](5_3_4%20多轮对话记忆管理/test_history_window.py)、[test_conversation_summary.py](5_3_4%20多轮对话记忆管理/test_conversation_summary.py) |
| 5_4_1 RAG与Agent深度融合 | 31 | [test_source_routing.py](5_4_1%20RAG与Agent深度融合/test_source_routing.py)、[test_memory_context.py](5_4_1%20RAG与Agent深度融合/test_memory_context.py) |
| 5_4_2 可观测性与健康检查 | 58 | [test_health_check.py](5_4_2%20可观测性与健康检查/test_health_check.py)、[test_agent_metrics.py](5_4_2%20可观测性与健康检查/test_agent_metrics.py) |
| 5_4_3 前端与会话管理 | 39 | [test_document_management.py](5_4_3%20前端与会话管理/test_document_management.py)、[test_chat_streaming.py](5_4_3%20前端与会话管理/test_chat_streaming.py)、[test_conversation_history.py](5_4_3%20前端与会话管理/test_conversation_history.py)、[test_agent_streaming.py](5_4_3%20前端与会话管理/test_agent_streaming.py)、[test_citation_page.py](5_4_3%20前端与会话管理/test_citation_page.py) |
| 5_4_4 端到端联调与测试 | 34 | [test_document_flow.py](5_4_4%20端到端联调与测试/test_document_flow.py) |
| 5_5_2 系统性能评估 | 10 | [test_system_evaluation.py](5_5_2%20系统性能评估/test_system_evaluation.py) |
| 5_5_3 Bad Case分析与优化 | 19 | [test_bad_case_regression.py](5_5_3%20Bad%20Case分析与优化/test_bad_case_regression.py) |

## 运行方式

在项目根目录执行全部测试：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

只运行一个课程小节（此例 41 项）：

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

## 样例与验证边界

- PDF／Word／文本样例由测试临时生成，加载／分块实际执行；Chroma、SQLite、线程和Streamlit AppTest使用真实临时资源。
- [helpers.py](helpers.py)复用明确二维模拟向量、NDJSON和日志隔离。HTTP模型为mock，模拟向量不代表M3E效果，自动回归不测真实模型质量或性能、不连接Windows。
- 需项目依赖及本地Qwen词表；Agent测试前后隔离日志，测试不写默认用户知识库／会话。
- 上传／索引／删除恢复AppTest归5.4.4，组件／聊天／会话UI归5.4.3，会话并发归5.3.4。专项入口按实际职责选用例，数量不能与全量相加。

旧test_retrieval.py／test_generation.py／test_agent.py已经拆分，不再作入口。历史迁移与633／654／668阶段日志保留在[报告目录](../reports/README.md)，无需在此重复累计。真实模型、浏览器与质量缺陷见[完整性说明](../docs/QA/完整性验证.md)，修复前后与提交见[Mac修复记录](../reports/5_5_4%20项目交付/Mac可修复问题记录_20261004.md)。
