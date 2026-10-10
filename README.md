# 智能科研助理

本科课程项目：基于 RAG + Agent 的论文知识库问答系统。支持 PDF、DOCX、TXT、Markdown 导入，混合检索与模型重排、带来源的流式回答、八个本地工具和多轮会话。

## 技术与启动

单个 Streamlit 应用直接调用业务模块；Ollama + Qwen2.5:7b 本地生成，M3E 向量化，Chroma 持久化，BM25 + 手写 RRF + BGE 重排，SQLite 保存会话。配置集中在 [config.yaml](config.yaml)。

先准备 Python 3.10.10、依赖、Ollama 模型和本地权重，再在项目根目录启动。完整安装、模型下载、离线和 Docker 步骤见[用户使用手册](docs/%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C.md)。

```bash
# macOS / Linux：已有虚拟环境时直接运行。
.venv/bin/python -m streamlit run src/frontend/app.py
```

```powershell
# Windows：已有虚拟环境时直接运行。
.\.venv\Scripts\python.exe -m streamlit run src/frontend/app.py --server.address 0.0.0.0 --server.port 8501
```

打开[本机页面](http://localhost:8501)，上传论文并导入后，在“科研对话”提问；“文档检索”查看原文块，“知识库”管理文件。局域网访问使用部署机器当前 IP。

## 课程交付

| 内容 | 入口 |
| --- | --- |
| 需求原文与报告格式 | [课程实践](docs/%E5%8D%97%E4%BA%AC%E5%86%9C%E4%B8%9A%E5%A4%A7%E5%AD%A6%E8%AF%BE%E7%A8%8B%E5%AE%9E%E8%B7%B5.docx)、[交付模板](docs/%E9%A1%B9%E7%9B%AE%E4%BA%A4%E4%BB%98%E6%A8%A1%E6%9D%BF.docx) |
| 七章节课程报告、分工与截图 | [课程报告](docs/%E8%AF%BE%E7%A8%8B%E6%8A%A5%E5%91%8A.md) |
| 架构、接口、配置与30项验收状态 | [技术设计](docs/%E6%8A%80%E6%9C%AF%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3.md) |
| 安装、使用、维护与常见问题 | [用户手册](docs/%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C.md) |
| 全部课程实验、分块/系统/Bad Case 报告 | [报告索引](reports/README.md) |
| 自动回归与真实模型验证边界 | [测试说明](tests/README.md) |

## 当前状态

2026-10-10按课程基本范围精简：删除定制CSS、回收站/会话归档恢复、旧RAG历史兼容和停机索引修复CLI；摘要每次只压缩一批，Observation只解析完整JSON，文献列表共用一个实现。生成仅保留必要的证据、引用和预算规范，索引状态按块数判断。日常测试从853项减为106项；Mac 106项回归（6.22秒）和10项真实流程（约135秒）全部通过，见[本轮记录](reports/5_4_4%20端到端联调与测试/基本功能精简验收_20261010/acceptance.json)。保留三种分块、混合检索/BGE、八工具、缓存和窗口/摘要记忆。默认16K上下文、1024输出Token。

Windows已同步并部署，106项模块回归和10项真实基本流程通过；最终引用Prompt另做实机问答复测，事实、来源及刷新恢复通过。启动进程退出、旧会话外键和引用提示问题已处理；版本与证据见[Windows验收记录](reports/5_4_4%20端到端联调与测试/Windows基本功能精简验收_20261010/本轮部署验收报告.md)。

正式论文集质量尚未全部通过：已有120条助手初评，用户审核0/120；10月9日8条定向复测为3通过、1缺引用、4失败。本轮合成资料基本测试不改写这些历史分数。中文查询英文排名按用户安排暂缓；成员资料待填，Word/PPT保留10月7日快照，待结论确定后重建。详细结果和限制见[系统评测报告](reports/%E7%B3%BB%E7%BB%9F%E8%AF%84%E6%B5%8B%E6%8A%A5%E5%91%8A.md)和[本轮修复](reports/Bad_Case分析报告.md#4-基本任务集中修复2026-10-10)。

本轮恢复至`1f40458`后，精简参考本地`/Users/rean/github/Agent-RAG`（`88147d2`），保留课程所需的真实本地模型链路。原始源码参考 [dsy1018/ai-chatbot 固定版本](https://github.com/dsy1018/ai-chatbot/tree/6979d7173ed0f92910a8571dd4ca1b4a171a2c49)，按课程要求在既定目录补充有界 ReAct、实验和交付；不把上游宣传或课程示例指标当实测成果。
