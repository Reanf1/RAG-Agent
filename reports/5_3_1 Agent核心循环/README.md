# 5.3.1 Agent核心循环

Thought、Action、Observation、有界ReAct及System Prompt验证。

本目录归档资料共16个。日期、首次失败、修正与复测名称保持原样；文件存在不代表验收通过。

[返回总索引](../README.md) · [技术设计与验收](../../docs/技术设计文档.md)

## 实现说明

[合并QA](../../docs/QA/5.3.1%20Agent核心循环.md)

## 运行脚本

- [verify_action.py](verify_action.py)：验证工具选择与实际执行
- [verify_react.py](verify_react.py)：验证观察、继续及循环终止
- [verify_thought.py](verify_thought.py)：验证单轮决策

## 报告、评测输入与原始结果

- [Action工具执行验证结果_20261002.json](Action工具执行验证结果_20261002.json)
- [AgentSystemPrompt验证结果_20261002.json](AgentSystemPrompt验证结果_20261002.json)
- [AgentSystemPrompt验证结果_20261002_复测.json](AgentSystemPrompt验证结果_20261002_复测.json)
- [ReAct循环验证结果_20261002.json](ReAct循环验证结果_20261002.json)
- [ReAct循环验证结果_20261002_复测.json](ReAct循环验证结果_20261002_复测.json)
- [Thought单轮规划验证结果_20261002.json](Thought单轮规划验证结果_20261002.json)

## 执行日志与页面记录

- [Action自动化测试_20261002.log](Action自动化测试_20261002.log)
- [AgentSystemPrompt单元测试_20261002.log](AgentSystemPrompt单元测试_20261002.log)
- [AgentSystemPrompt格式约束缺陷复现_20261002.log](AgentSystemPrompt格式约束缺陷复现_20261002.log)
- [AgentSystemPrompt自动化测试_20261002.log](AgentSystemPrompt自动化测试_20261002.log)
- [AgentSystemPrompt自动化测试_20261002_复测.log](AgentSystemPrompt自动化测试_20261002_复测.log)
- [ReAct自动化测试_20261002.log](ReAct自动化测试_20261002.log)
- [Thought自动化测试_20261002.log](Thought自动化测试_20261002.log)
