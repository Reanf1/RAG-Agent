# 测试说明

日常测试保留106项，按课程模块覆盖四格式加载、三种分块、增量入库、混合检索/BGE、引用/缓存/降级、Agent工具/路由/并行/有限恢复、窗口/摘要、会话隔离与前端。删除取消功能、重复协议、样式和极端组合测试；共享样例见[helpers.py](helpers.py)。

```bash
# 在项目根目录运行；日常修改只跑相关模块，跨模块修改集中验证。
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

```bash
# 先启动Ollama并准备本地权重；使用临时知识库，验证八工具和页面基本流程。
.venv/bin/python 'reports/5_5_3 Bad Case分析与优化/verify_basic_tasks.py' --output /tmp/rag-basic-new
```

```powershell
# Windows运行入口；本次版本仅完成Mac验收。
.\.venv\Scripts\python.exe reports\模块完整性验证\verify_completeness_tests.py --scope all --output "$env:TEMP\rag-basic-new.json"
```

Mac 106项回归（6.22秒）与10项真实流程（约135秒）全部通过，结果、原答及失败处理见[验收记录](../reports/5_4_4%20端到端联调与测试/基本功能精简验收_20261010/acceptance.json)。单元测试使用临时Chroma/SQLite/AppTest，模型调用部分为mock；真实流程使用本地Qwen、M3E和BGE。合成资料用于确认功能，不能代替论文答案质量审核。

正式12篇/60题评测、课程性能对照与三个研究问题按[课程实验记录](../reports/课程实验记录.md)单独复现，历史JSON/日志保留；旧版界面核验脚本按其记录的Git版本运行，当前基本验收使用上述入口。当前Windows和实机浏览器尚未复测，Word/PPT暂不更新。
