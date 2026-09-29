# 测试计划

当前 `test_retrieval.py` 有 27 个测试：PDF 11 个、Word 8 个、TXT/Markdown 8 个。使用临时生成的实际文件验证中文、来源与位置、稳定文档标识、Word 正文/表格顺序、UTF-8 BOM、Markdown 保真及导入失败。使用标准库 `unittest`，不需要额外测试依赖。

在项目根目录执行：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_retrieval.py' -v
```

也支持从任意目录直接运行测试文件。使用已安装 `requirements.txt` 依赖的 Python，例如项目虚拟环境：

```bash
/Users/rean/github/RAG+Agent/.venv/bin/python /Users/rean/github/RAG+Agent/tests/test_retrieval.py -v
```

测试入口会按 `__file__` 定位项目根目录，避免直接运行时找不到 `src`。

新增的 Word 测试需要 `python-docx`。依赖已安装到项目 `.venv`；若使用其他 Python 解释器，应先为该解释器安装 `requirements.txt`，仅更换启动路径不会共享虚拟环境的依赖。

其他业务仍为占位，不为占位代码编写无意义的测试。

实现后按以下顺序添加有明确预期的用例：

1. 文档加载：PDF 页码、Word 表格、中文编码与批量导入失败。
2. 分块与检索：边界、元数据保留、RRF 去重、增量索引及检索指标。
3. 生成：引用准确性、流式输出、缓存失效与三类降级场景。
4. Agent：工具选择、并行、超时、最大循环次数、会话隔离与记忆截断。
5. 端到端：上传 → 索引 → 提问 → 决策 → 带引用回答；空库、大文档、长对话。

后续课程要求的检索与 Agent 用例分别放入 `test_retrieval.py`、`test_agent.py`；实际采用的测试依赖在首次使用时加入。
