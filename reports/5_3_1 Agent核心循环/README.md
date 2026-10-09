# 5.3.1 Agent核心循环

> 状态口径（2026-10-09）：当前实现与验收以[技术设计](../../docs/%E6%8A%80%E6%9C%AF%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3.md)为准。带日期的实验数值、截图和“本轮／待完成”描述属于对应历史阶段；自动回归、真实链路、助手初评与用户审核分别记录，局部复测不替代完整题集。Word／PPT保留10月7日快照，按用户安排待结论确定后重建。

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
