# 测试说明

按课程5.1.1–5.4.4小节组织测试，共享样例见[helpers.py](helpers.py)。在项目根目录执行，结果输出使用新路径。

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

2026-10-09业务补丁2c0353c：macOS与Windows各853项通过，无失败/跳过；分别59.191秒、315.808秒。记录：[本地回归](../reports/5_5_3%20Bad%20Case%E5%88%86%E6%9E%90%E4%B8%8E%E4%BC%98%E5%8C%96/%E6%9C%80%E7%BB%88%E8%B4%A8%E9%87%8F%E6%94%B6%E5%B0%BE_20261009/%E6%9C%AC%E5%9C%B0%E5%85%A8%E9%87%8F%E5%9B%9E%E5%BD%92_%E4%BF%AE%E5%A4%8D%E5%90%8E.json)、[Windows回归](../reports/5_5_3%20Bad%20Case%E5%88%86%E6%9E%90%E4%B8%8E%E4%BC%98%E5%8C%96/%E6%9C%80%E7%BB%88%E8%B4%A8%E9%87%8F%E6%94%B6%E5%B0%BE_20261009/Windows%E8%A1%A5%E4%B8%81%E5%A4%8D%E6%B5%8B/regression.json)。

加载/分块、Chroma、SQLite、线程与Streamlit AppTest使用真实临时资源；模型HTTP与部分向量使用mock，不能说明真实M3E/BGE/Ollama质量或性能。需项目依赖和本地Qwen计数词表；测试隔离日志和用户数据库。

真实模型、浏览器、并发和容器另见[课程实验记录](../reports/%E8%AF%BE%E7%A8%8B%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95.md)。最新56次真实并发0异常用于验证线程/分词器修复，不是吞吐量实验；8条定向问答仍有缺引用/语义失败，见[Bad Case报告](../reports/Bad_Case%E5%88%86%E6%9E%90%E6%8A%A5%E5%91%8A.md)。
