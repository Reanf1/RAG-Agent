# 智能科研助理

基于 RAG + Agent 的论文知识库问答系统，依据南京农业大学生产实习课程要求建设。

当前已实现 PDF、Word（.docx）及 TXT/Markdown 加载，以及 Streamlit 批量上传、进度显示、状态追踪和失败项重试。固定大小、递归字符、句段边界三种分块策略已实现；共 79 个测试通过。PDF 已补常见双栏排序、表格与续表处理、公式符号/上下标及原文定位。导入成功表示原始文件已保存、正文已加载；页面导入尚未接入分块，索引、检索、生成与 Agent 仍待实现。实现说明见 [PDF 加载器选择](docs/QA/5.1.1%20PDF加载器选择.md)、[Word 和纯文本加载](docs/QA/5.1.1%20Word与纯文本加载器实现.md)、[批量文档导入](docs/QA/5.1.1%20批量文档导入与状态追踪.md)、[学术 PDF 解析优化](docs/QA/5.1.2%20学术论文PDF解析优化.md) 与 [三种文本分块策略](docs/QA/5.1.2%20三种文本分块策略.md)。

## 课程依据

- [课程实践方案](docs/南京农业大学课程实践.docx)：项目功能、实验要求与交付物。
- [项目交付模板](docs/项目交付模板.docx)：课程报告章节、格式和目录示例。
- [技术设计与验收](docs/技术设计文档.md)：固定目录、参考代码映射、五模块需求和验收方式。

默认采用课程主方案“智能科研助理”。企业制度、教学辅导、法律文书三个题目为备选，不纳入当前骨架。

## 快速启动

当前项目 `.venv` 使用 Python 3.10.10，根目录 `.python-version` 同步指定该版本，供 pyenv 选择解释器。重建环境前确认 `python3.10 --version` 输出 `Python 3.10.10`。

```bash
# 在项目根目录创建并启用虚拟环境。
python3.10 -m venv .venv
source .venv/bin/activate

# 安装当前页面、文档加载与分块需要的依赖。
python -m pip install -r requirements.txt

# 启动文档导入应用。
python -m streamlit run src/frontend/app.py
```

打开 http://localhost:8501，在左侧选择多份文档并点击“开始导入”，查看进度、状态和失败原因；“重试失败项”只重新处理失败文件。单份文件默认最大 20 MB，可在 `config.yaml` 的 `importing.max_file_size_mb` 调整。问答尚未接入。

`requirements.txt` 只声明当前实际使用的依赖。后续实现文档处理、向量检索和模型调用时，再加入对应依赖并验证版本。

## 目录结构

```text
RAG+Agent/
├── AGENTS.md                  开发约定
├── README.md                  项目入口
├── requirements.txt           当前运行依赖
├── config.yaml                页面、分块配置与后续模块参数样例
├── src/
│   ├── data_loader/            模块一：PDF、Word、TXT/Markdown 加载
│   ├── chunking/               模块一：固定、递归、语义分块
│   ├── retrieval/              模块一：向量、BM25、RRF、重排序
│   ├── generation/             模块二：Prompt、RAG、流式、缓存
│   ├── agent/                  模块三：ReAct、工具、路由、记忆
│   ├── frontend/               模块四：Streamlit 入口、页面与组件
│   └── utils/                  配置读取与日志预留位置
├── tests/                     文档加载、分块、批量导入及界面测试
├── data/                      原始文献与索引，内容默认不提交
├── logs/                      运行日志，内容默认不提交
├── docs/                      原始资料、需求、技术设计、使用手册、课程报告
├── reports/                   评测集、系统评测、分块实验与 Bad Case 报告
└── docker/                    Dockerfile 与 Compose 文件
```

## 实施顺序

1. 文档加载 → 三种分块 → 增量索引 → 向量/BM25/RRF/重排序；核验召回与页码溯源。
2. RAG 生成 → 引用 → 流式 → 缓存与降级；核验回答来源和边缘场景。
3. 手写 ReAct → 工具集 → 路由/并行/恢复 → 会话记忆；核验工具选择和终止条件。
4. 前端联调 → 监控与健康检查；核验上传论文到得到回答的完整流程。
5. 10–20 篇论文、至少 50 条评测问题 → 对比实验 → Bad Case 优化 → 完整交付。

模型与存储的初步方案为 Ollama + Qwen2.5、BGE Embedding、Chroma、Streamlit；这是按课程方案选取的开发起点，尚未完成模型安装和选型实验。配置数值均为实验起点，不代表最优结果。

在项目根目录调用分块模块：

```python
from src.data_loader import load_document
from src.chunking import split_documents

documents = load_document("data/raw/论文.pdf")  # 改为实际文档路径。
chunks = split_documents(documents)  # 读取 config.yaml：recursive、512/64 字符。
chunks = split_documents(documents, strategy="semantic")  # 句段策略使用 YAML 默认参数。
```

策略名为 `fixed`、`recursive`、`semantic`。仅固定策略比较 256/512/1024 字符；递归和句段策略使用 YAML 默认的 512/64 参数。正文块不超过设置大小；已独立识别的 PDF/Word 表格整块保留，可能超限。块继承来源位置并增加稳定 ID 与字符区间。语义策略基于句段规则，重叠只复用完整单元，可能为零；不使用语义模型。当前仅记录 [分块统计](reports/分块策略对比实验报告.md)，召回实验待检索层接入。

## 交付文档

- [技术设计文档](docs/技术设计文档.md)
- [用户使用手册](docs/用户使用手册.md)
- [课程报告骨架](docs/课程报告.md)
- [系统评测报告](reports/系统评测报告.md)
- [分块策略对比实验报告](reports/分块策略对比实验报告.md)
- [Bad Case 分析报告](reports/Bad_Case分析报告.md)

Docker 启动：`docker compose -f docker/docker-compose.yml up --build`。部署入口为同一 Streamlit 应用，当前未实测容器中的导入功能；后续接入模型与索引服务后再补充完整系统的部署验证。
