# 智能科研助理

基于 RAG + Agent 的论文知识库问答系统，依据南京农业大学生产实习课程要求建设。

当前已实现 PDF、Word（.docx）及 TXT/Markdown 加载，以及 Streamlit 批量上传、进度显示、状态追踪和失败项重试。固定、递归、句段三种分块、本地 Embedding 和 Chroma 操作封装已实现；上传后自动分块、批量向量化并增量构建索引，页面支持向量、BM25、RRF 与模型重排 Top-K 检索，共 238 个测试通过；RAG Prompt、上下文排序/截断、引用溯源、本地流式/非流式生成和页面问答已实现，八组生成参数对照已完成。PDF 已补常见双栏排序、表格与续表处理、公式符号/上下标及原文定位。Embedding 经人工智能论文中英双语基准重新比较后保留 M3E-base；跨语言质量仍不足。导入成功表示原文已保存、全部预期块已处理并写入索引；重复块跳过，新文献追加，语义缓存、完整降级、日志与 Agent 仍待实现。实现说明见 [PDF 加载器选择](docs/QA/5.1.1%20PDF加载器选择.md)、[Word 和纯文本加载](docs/QA/5.1.1%20Word与纯文本加载器实现.md)、[批量文档导入](docs/QA/5.1.1%20批量文档导入与状态追踪.md)、[学术 PDF 解析优化](docs/QA/5.1.2%20学术论文PDF解析优化.md)、[三种文本分块策略](docs/QA/5.1.2%20三种文本分块策略.md)、[Embedding 对比和选择](docs/QA/5.1.3%20Embedding模型对比与选择.md)、[向量数据库对比和选择](docs/QA/5.1.3%20向量数据库对比与选择.md)、[批量向量化及增量索引](docs/QA/5.1.3%20批量向量化与增量索引.md)、[向量 Top-K 检索](docs/QA/5.1.3%20向量相似度Top-K检索.md)、[BM25 关键词检索](docs/QA/5.1.4%20BM25关键词检索.md)、[RRF 混合检索](docs/QA/5.1.4%20RRF混合检索.md)、[模型重排序](docs/QA/5.1.4%20模型重排序.md) 与 [三档检索评测](docs/QA/5.1.4%20检索质量评估实验.md)，生成参数见 [对比与选择](docs/QA/5.2.1%20生成参数对比与选择.md)，动态引用和真实页面验证见 [流式 QA](docs/QA/5.2.2%20流式输出与引用.md)。

当前 6 篇人工智能论文、96 条中英配对查询的三档实测已完成：Hit@5 为 34.38% / 35.42% / 45.83%，MRR@5 为 0.2188 / 0.2597 / 0.3788。原始排名、三轮计时、图表及 Excel 见 [实验报告](docs/QA/5.1.4%20检索质量评估实验.md)。这是 48 个意图的开发集，标注尚未经独立人工复核，不替代完整系统评测。

模块一完整性核验：165 项测试与真实本地模型链路通过；五组分块设置的检索召回对比仍缺失，模块一保持部分完成。详见 [完整性验证报告](docs/QA/5.1%20模块一完整性验证.md)。

RAG 专用 Prompt 已完成系统角色、检索上下文、用户问题和输出格式四部分，当前版本 `rag-v2`；返回 LangChain 消息，约束真实来源引用和资料不足提示。调用与验证见 [Prompt QA](docs/QA/5.2.1%20RAG专用Prompt模板.md)。已补按相关性拼接和动态字符预算截断，调用与边界见 [上下文 QA](docs/QA/5.2.1%20上下文智能拼接与截断.md)；已补按正文编号展示文件名/页码和原文证据的 [引用溯源接口](docs/QA/5.2.1%20引用溯源机制.md)；本地模型和页面流式问答已接通，独立引用语义评测仍待完成。

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

# 安装页面、文档加载、分块、本地 Embedding 与 Chroma 依赖。
python -m pip install -r requirements.txt

# 按用户手册准备本地 M3E 权重后，启动文档导入/索引应用。
python -m streamlit run src/frontend/app.py
```

打开 http://localhost:8501，先按 [用户手册](docs/用户使用手册.md) 准备本地 M3E 权重，在左侧选择多份文档并点击“开始导入”，查看加载/分块/索引阶段、分块数和本次新增数。新文献增量加入；“重试失败项”只处理失败文件，索引失败时复用已加载原文并跳过已写入块。单份文件默认最大 20 MB，可在 `config.yaml` 的 `importing.max_file_size_mb` 调整。提问前按手册启动本地 Ollama，“RAG 流式问答”默认使用混合检索和模型重排，逐步输出正文并在引用位置补全文档名/真实位置；来源证据可展开。当前历史只保留页面展示。

`requirements.txt` 只声明当前实际使用的依赖。后续功能需要新依赖时，再加入并验证版本。

## 目录结构

```text
RAG+Agent/
├── AGENTS.md                  开发约定
├── README.md                  项目入口
├── requirements.txt           当前运行依赖
├── config.yaml                页面、分块、Embedding 配置与后续参数样例
├── src/
│   ├── data_loader/            模块一：PDF、Word、TXT/Markdown 加载
│   ├── chunking/               模块一：固定、递归、语义分块
│   ├── retrieval/              模块一：M3E/Chroma/BM25/RRF/模型重排已实现
│   ├── generation/             模块二：Prompt、RAG、流式、缓存
│   ├── agent/                  模块三：ReAct、工具、路由、记忆
│   ├── frontend/               模块四：Streamlit 入口、页面与组件
│   └── utils/                  配置读取与日志预留位置
├── tests/                     文档加载、分块、Embedding、Chroma、批量导入及界面测试
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

当前 Embedding 选择本地 `moka-ai/m3e-base`，基于 3 篇中文、3 篇英文人工智能论文重新比较两模型后保留，权重已在本机准备。宏平均 Hit@5 为 M3E 34.38%、BGE 31.25%，M3E 文档编码约快 3.14 倍；英文与跨语言检索仍有明显不足，选型不表示质量已达标。向量库经 Chroma/FAISS 实测后选择 Chroma，`search_ef=100`；本地 Ollama + Qwen2.5:7b 流式/非流式接口、参数实验和 RAG 页面已接通；完整 Agent 系统联调待实现。配置数值均为工程起点，不代表全部科研场景的最优结果。

在项目根目录调用分块模块：

```python
from src.data_loader import load_document
from src.chunking import split_documents

documents = load_document("data/raw/论文.pdf")  # 改为实际文档路径。
chunks = split_documents(documents)  # 读取 config.yaml：recursive、512/64 字符。
chunks = split_documents(documents, strategy="semantic")  # 句段策略使用 YAML 默认参数。
```

策略名为 `fixed`、`recursive`、`semantic`。仅固定策略比较 256/512/1024 字符；递归和句段策略使用 YAML 默认的 512/64 参数。正文块不超过设置大小；已独立识别的 PDF/Word 表格整块保留，可能超限。块继承来源位置并增加稳定 ID 与字符区间。语义策略基于句段规则，重叠只复用完整单元，可能为零；不使用语义模型。当前仅记录 [分块统计](reports/分块策略对比实验报告.md)，召回实验待检索层接入。

已准备本地权重后，可继续生成向量：

```python
from src.retrieval.vector_store import get_embeddings

embeddings = get_embeddings()  # 首次加载，之后在进程内复用；只读本地 M3E。
vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
query_vector = embeddings.embed_query("论文使用了什么实验方法？")
english_vector = embeddings.embed_query("Which experimental method does the paper use?")
print(len(vectors), len(query_vector))  # 查询向量为 768 维，向量已归一化。
```

首次准备权重的操作见 [用户手册](docs/用户使用手册.md)。Embedding 的 821 个论文分块、96 条查询（48 个中英配对意图）实测与原始排名/计时见 [QA](docs/QA/5.1.3%20Embedding模型对比与选择.md)、[双语评测集](reports/Embedding论文双语评测集.json) 和 [结果 JSON](reports/Embedding论文双语对比结果.json)。问题与标注由 Codex 根据原文编写，尚未经人工复核；旧中文网页数据保留在原结果文件中，不替代论文知识库完整系统评测。

也可将分块持久化并检索：

```python
from src.retrieval.vector_store import VectorStore

store = VectorStore()  # 打开本地 Chroma；新增块或有效向量查询时才加载 M3E。
print("新增块数：", store.add_chunks(chunks))  # 重复导入跳过已有 chunk_id。
for document, score in store.search("论文使用了什么方法？"):
    print(document.metadata["source_file"], score, document.page_content)
print("库中块数：", store.count())
# 需要移除某份文档的索引时调用；原始文件保留。
# store.delete_document(chunks[0].metadata["doc_id"])
```

两库实测、默认参数不足与最终选择见 [向量数据库 QA](docs/QA/5.1.3%20向量数据库对比与选择.md)。FAISS 仅用于独立实验，不是业务运行依赖。上传页面已自动构建索引，也可通过 `batch_build_index()` 完成多文件入库，见 [批量索引 QA](docs/QA/5.1.3%20批量向量化与增量索引.md)。

页面中的“文档 Top-K 检索”可选择向量相似度、BM25 关键词、RRF 混合检索或 RRF + 模型重排，输入中英文问题、设置 K（默认 5），并可填写文档 ID 限定范围。四种方式均按各自分数降序展示正文、文件名和来源位置；分数不可直接比较，也不是命中概率。向量接口为 `store.search(query, k=5, doc_id=None)`；关键词接口为 `BM25Retriever().search(query, k=5, doc_id=None)`。BM25 从 Chroma 正文重建内存索引，无需 Embedding 权重；Python 长期复用实例时，在新增/删除后调用 `rebuild()`，页面每次提交自动读取最新语料。英文按词、中文按单字，支持全角字符归一化；没有词项交集返回空列表，命中结果保留零分/负分。说明见 [Top-K QA](docs/QA/5.1.3%20向量相似度Top-K检索.md) 与 [BM25 QA](docs/QA/5.1.4%20BM25关键词检索.md)。当前返回片段，尚不生成答案。

混合接口为 `HybridRetriever().search(query, k=5, doc_id=None)`：默认两路各召回 20 项，按 `chunk_id` 合并并累加 `1/(60+排名)`，排名从 1 开始，再取最终 K 项。BM25 每次读取当前正文；新增/删除后再查即可，不重算旧文档向量。有效混合查询需本地 M3E，错误明确上报。算法与核验见 [RRF QA](docs/QA/5.1.4%20RRF混合检索.md)。

启用精排调用 `HybridRetriever().search(query, k=5, rerank=True)`，先将融合后的默认 Top-20 交给本地 `bge-reranker-base`，再按模型分数取最终 K 项。页面选择“RRF + 模型重排”；本机权重已准备，首次有效精排才加载，失败不改用关键词。复用现有 sentence-transformers，不新增依赖；权重准备、Token 截断与真实核验见 [模型重排 QA](docs/QA/5.1.4%20模型重排序.md)。

## 交付文档

- [技术设计文档](docs/技术设计文档.md)
- [用户使用手册](docs/用户使用手册.md)
- [课程报告骨架](docs/课程报告.md)
- [系统评测报告](reports/系统评测报告.md)
- [分块策略对比实验报告](reports/分块策略对比实验报告.md)
- [Bad Case 分析报告](reports/Bad_Case分析报告.md)

Docker 启动：`docker compose -f docker/docker-compose.yml up --build`。部署入口为同一 Streamlit 应用，当前未实测容器中的导入功能；后续接入模型与索引服务后再补充完整系统的部署验证。
