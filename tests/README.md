# 测试计划

当前 `test_retrieval.py` 有 58 个测试：基础 PDF 11 个、学术 PDF 17 个、Word 8 个、TXT/Markdown 8 个、批量导入 12 个、Streamlit 界面 2 个。使用临时生成的实际文件验证中文、来源与位置、稳定文档标识、Word 正文/表格顺序、UTF-8 BOM 和 Markdown 保真。

学术 PDF 用例通过真实绘制文字、横线、网格和图片，验证双栏/通栏顺序、单栏防误判、两页与三页续表、重复表头去重、中文续表、无编号或不兼容表格防误合并、网格/三线表共存、空单元格、密集公式行、复杂表头回退、数学符号/上下标及图像原文裁剪。它们不依赖网络下载；公开论文的人工核对另记录在 [学术 PDF 解析优化](../docs/QA/5.1.2%20学术论文PDF解析优化.md)。

批量用例验证多格式导入、进度与状态、失败不阻断后续文件、仅重试失败项、保存失败恢复、同名文件保护、重复项和大小限制。界面用例通过 Streamlit AppTest 实际操作上传组件与按钮，检查状态表、进度条和页面重跑；只替换测试保存目录，并模拟临时错误以核验重试成功。所有文件使用临时目录，不修改真实文献。

测试使用标准库 `unittest` 和已安装 Streamlit 自带的 AppTest，不增加测试依赖；不等同于浏览器视觉检查。

AppTest 等待上限为 10 秒，兼顾新环境首次加载界面依赖的耗时，避免默认 3 秒等待误报。

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

1. 分块与检索：边界、元数据保留、RRF 去重、增量索引及检索指标。
2. 生成：引用准确性、流式输出、缓存失效与三类降级场景。
3. Agent：工具选择、并行、超时、最大循环次数、会话隔离与记忆截断。
4. 端到端：上传 → 索引 → 提问 → 决策 → 带引用回答；空库、大文档、长对话。

后续课程要求的检索与 Agent 用例分别放入 `test_retrieval.py`、`test_agent.py`；实际采用的测试依赖在首次使用时加入。
