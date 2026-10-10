# 测试说明

按课程5.1.1–5.4.4小节组织测试，共享样例见[helpers.py](helpers.py)。在项目根目录执行，结果输出使用新路径。

日常修改先运行相关模块；基本验收用下方10项真实流程，覆盖上传入库、八个工具、多轮对话和页面保存。跨模块改动再集中跑全量回归。已有故障回归保留，不继续扩展极端输入组合；正式12篇/60题评测按课程实验单独执行。

```bash
# 先启动本地Ollama并备齐权重；脚本使用临时知识库，不改现有数据。
.venv/bin/python 'reports/5_5_3 Bad Case分析与优化/verify_basic_tasks.py' --output /tmp/rag-basic-new
```

```bash
# macOS / Linux全量回归
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
# 单模块示例
.venv/bin/python 'tests/5_3_1 Agent核心循环/test_action.py' -v
```

```powershell
# Windows：JSON运行器处理跨盘临时目录与Chroma句柄清理。
.\.venv\Scripts\python.exe reports\模块完整性验证\verify_completeness_tests.py --scope all --output "$env:TEMP\rag-regression-new.json"
```

macOS也可使用同一JSON运行器；`--scope module4`用于模块四专项，数量不与全量相加。项目外执行须使用测试文件及虚拟环境的绝对路径。

## 结果与边界

2026-10-10精简后：Mac 上 850 项自动回归通过（59.304秒，无失败/错误/跳过），10 项真实基本流程通过。删除14个旧算法测试，主要是章节/指标固定句式与邻块选择规则；未按数量目标删除功能测试。见[本轮原答与回归](../reports/5_4_4%20端到端联调与测试/Mac精简验收_20261010/acceptance.json)。Windows未部署本轮精简代码。

精简前2026-10-10集中验收：本地864项回归通过（58.837秒），Windows 上 864 项通过（306.444秒），无失败、错误或跳过。Windows 上 10 项真实流程覆盖八个本地工具及AppTest；实机浏览器另测完整问答、长文件名关键词、侧栏状态和刷新恢复。见[回归与原答](../reports/5_4_4%20端到端联调与测试/Windows基本功能验收_20261010/acceptance.json)。合成资料检验基本流程，不代表正式论文集质量全部通过。

2026-10-09业务补丁2c0353c：macOS与Windows各853项通过，无失败/跳过；分别59.191秒、315.808秒。记录：[本地回归](../reports/5_5_3%20Bad%20Case%E5%88%86%E6%9E%90%E4%B8%8E%E4%BC%98%E5%8C%96/%E6%9C%80%E7%BB%88%E8%B4%A8%E9%87%8F%E6%94%B6%E5%B0%BE_20261009/%E6%9C%AC%E5%9C%B0%E5%85%A8%E9%87%8F%E5%9B%9E%E5%BD%92_%E4%BF%AE%E5%A4%8D%E5%90%8E.json)、[Windows回归](../reports/5_5_3%20Bad%20Case%E5%88%86%E6%9E%90%E4%B8%8E%E4%BC%98%E5%8C%96/%E6%9C%80%E7%BB%88%E8%B4%A8%E9%87%8F%E6%94%B6%E5%B0%BE_20261009/Windows%E8%A1%A5%E4%B8%81%E5%A4%8D%E6%B5%8B/regression.json)。

加载/分块、Chroma、SQLite、线程与Streamlit AppTest使用真实临时资源；模型HTTP与部分向量使用mock，不能说明真实M3E/BGE/Ollama质量或性能。需项目依赖和本地Qwen计数词表；测试隔离日志和用户数据库。

真实模型、浏览器、并发和容器另见[课程实验记录](../reports/%E8%AF%BE%E7%A8%8B%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95.md)。最新56次真实并发0异常用于验证线程/分词器修复，不是吞吐量实验；8条定向问答仍有缺引用/语义失败，见[Bad Case报告](../reports/Bad_Case%E5%88%86%E6%9E%90%E6%8A%A5%E5%91%8A.md)。
