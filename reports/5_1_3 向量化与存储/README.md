# 5.1.3 向量化与存储

Embedding中英双语对比、Chroma/FAISS选型、增量索引和向量Top-K验证。

本目录归档资料共11个。日期、首次失败、修正与复测名称保持原样；文件存在不代表验收通过。

[返回总索引](../README.md) · [技术设计与验收](../../docs/技术设计文档.md)

## 实现说明

[合并QA](../../docs/QA/5.1.3%20向量化与存储.md)

## 运行脚本

- [compare_embeddings.py](compare_embeddings.py)：比较两个Embedding的效果与速度
- [compare_vector_stores.py](compare_vector_stores.py)：比较Chroma与FAISS
- [prepare_paper_benchmark.py](prepare_paper_benchmark.py)：准备真实中英文论文开发集

## 报告、评测输入与原始结果

- [Embedding对比结果.json](Embedding对比结果.json)
- [Embedding数值复核.json](Embedding数值复核.json)
- [Embedding论文双语对比结果.json](Embedding论文双语对比结果.json)
- [Embedding论文双语评测集.json](Embedding论文双语评测集.json)
- [向量Top-K检索验证结果.json](向量Top-K检索验证结果.json)
- [向量数据库对比结果.json](向量数据库对比结果.json)
- [向量数据库默认参数对比结果.json](向量数据库默认参数对比结果.json)
- [批量索引验证结果.json](批量索引验证结果.json)
