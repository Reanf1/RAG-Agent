# 智能科研助理

基于 RAG + Agent 的论文知识库问答系统，依据南京农业大学生产实习课程要求建设。

当前已完成项目初始化和 PyMuPDF 分页文本加载器，支持来源与页码元数据，已通过文档加载测试。其他格式加载、检索、生成与 Agent 等业务模块仍为占位；前端尚未接入上传。PDF 选型依据见 [PDF 加载器选择](docs/QA/5.1.1%20PDF加载器选择.md)。

## 课程依据

- [课程实践方案](docs/南京农业大学课程实践.docx)：项目功能、实验要求与交付物。
- [项目交付模板](docs/项目交付模板.docx)：课程报告章节、格式和目录示例。
- [技术设计与验收](docs/技术设计文档.md)：固定目录、参考代码映射、五模块需求和验收方式。

默认采用课程主方案“智能科研助理”。企业制度、教学辅导、法律文书三个题目为备选，不纳入当前骨架。

## 快速启动

Python 3.10 及以上；本次初始化使用 Python 3.12 验证。

```bash
# 在项目根目录创建并启用虚拟环境。
python3 -m venv .venv
source .venv/bin/activate

# 安装当前页面与 PDF 加载器需要的依赖。
python -m pip install -r requirements.txt

# 启动项目说明页。
python -m streamlit run src/frontend/app.py
```

打开 http://localhost:8501。页面展示项目范围和模块状态，目前不提供文档上传或问答。

`requirements.txt` 只声明当前实际使用的依赖。后续实现文档处理、向量检索和模型调用时，再加入对应依赖并验证版本。

## 目录结构

```text
RAG+Agent/
├── AGENTS.md                  开发约定
├── README.md                  项目入口
├── requirements.txt           当前运行依赖
├── config.yaml                页面配置与后续模块参数样例
├── src/
│   ├── data_loader/            模块一：PDF、Word、TXT/Markdown 加载
│   ├── chunking/               模块一：固定、递归、语义分块
│   ├── retrieval/              模块一：向量、BM25、RRF、重排序
│   ├── generation/             模块二：Prompt、RAG、流式、缓存
│   ├── agent/                  模块三：ReAct、工具、路由、记忆
│   ├── frontend/               模块四：Streamlit 入口、页面与组件
│   └── utils/                  配置与日志的预留位置
├── tests/                     后续有实际实现后添加测试
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

## 交付文档

- [技术设计文档](docs/技术设计文档.md)
- [用户使用手册](docs/用户使用手册.md)
- [课程报告骨架](docs/课程报告.md)
- [系统评测报告](reports/系统评测报告.md)
- [分块策略对比实验报告](reports/分块策略对比实验报告.md)
- [Bad Case 分析报告](reports/Bad_Case分析报告.md)

Docker 启动：`docker compose -f docker/docker-compose.yml up --build`。当前镜像只运行初始化页面，后续接入模型与索引服务后再补充完整系统的部署验证。
