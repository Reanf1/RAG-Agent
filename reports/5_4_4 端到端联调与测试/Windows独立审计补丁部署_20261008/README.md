# Windows 独立审计补丁部署与复测

> 状态口径（2026-10-09）：当前实现与验收以[技术设计](../../../docs/%E6%8A%80%E6%9C%AF%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3.md)为准。带日期的实验数值、截图和“本轮／待完成”描述属于对应历史阶段；自动回归、真实链路、助手初评与用户审核分别记录，局部复测不替代完整题集。Word／PPT保留10月7日快照，按用户安排待结论确定后重建。

2026-10-08，已完成 Windows 原生部署、全量回归、真实页面、模型和独立容器离线复测。现有原生页面保持运行：[智能科研助理](http://192.168.31.23:8501/)。项目为 `E:\工作\RAG-Agent`，业务源码基准 `c90ed95732bff1e272b5d411a2ab2fd8d9cb3ef0`。本报告提交只补证据与验证脚本，业务源码保持该基准。

用户确认 IP 从 `192.168.1.12` 改为 `192.168.31.23` 后恢复 SSH；新地址使用已知旧地址的同一主机密钥校验，不绕过主机认证。旧地址失败及初次不完整指纹仍保留，均不计为最终结果。未保存登录密码。

## 核验结果

| 项目 | 本轮真实结果 | 证据 |
| --- | --- | --- |
| Git 同步与源码 | 从 `d6bdde4` 快进至 `c90ed95`；119/119 文件一致，仅归一 LF/CRLF，另存实际字节 SHA | [部署指纹](deployed-fingerprint.json)、[同步日志](Windows同步与原库副本.log) |
| Windows 全量回归 | 815/815，55 文件、18 小节；失败/错误/跳过均 0，207.838 秒为测试运行时间 | [逐项结果](regression-815.json)、[完整日志](Windows全量回归.log) |
| 默认编码与依赖 | Python 3.10.10，默认 cp936、UTF-8 模式 0；只将 stdout 设置 UTF-8；现有依赖 `pip check` 通过，未整体升级 | [预检](Windows只读预检.log)、[同步日志](Windows同步与原库副本.log) |
| 真实模型与缓存 | M3E/BGE 各三线程仅构造一次；9 对条件样例符合预期；Qwen TXT 问答、精确/语义命中、文档/配置/同数量正文失效通过 | [真实模型结果](real-runtime.json) |
| 原文重新建库 | 六原件 SHA 不变；新 276 块/6 篇，独立进程校验持久化向量和原生 HNSW 后启用 | [建库](reparse-build.json)、[重开验证](reparse-verify.json)、[切换](reparse-activate.json) |
| 页面与进程重启 | Markdown 代号和行号、ViT 英文回答及原页预览可用；刷新/进程重启恢复问答、轨迹和 Token；页面显示 6/6/276 | [页面记录](页面进程重启恢复.txt)、[重启后知识库](页面知识库重启后.jpg)、[进程重启](native-restart.json) |
| 原数据保护 | 旧 278 块内容指纹及六原件一致；61 个原会话、194 条消息、1 份摘要、5 条 RAG 历史逐行全部保留 | [旧库核验](old-backup-verified.json)、[重启后只读核验](native-state-after-restart.json) |
| Windows 容器 | 最新 linux/amd64 镜像构建通过；36 个 Python 源文件一致，CPU torch 2.5.1、SQLite 3.46.1；Chroma/SQLite 重启恢复，真实 M3E/BGE/Qwen 回答并引用 | [构建](docker-build-result.json)、[种子与重启](container-seed.json)、[真实容器运行](container-verify.json) |
| 公网隔离与收尾 | app/ollama 仅连接 internal 网络；直接公共 IP 443 访问失败，模型容器返回 Network is unreachable；三个服务重启后 healthy、无 OOM。仅移除本轮临时容器/网络，原服务保留 | [容器验证](container-verify.json)、[收尾状态](final-state-precommit.json) |

[验证汇总](验证汇总.json)由以上实际结果生成。回归中的受控 HTTP、小型向量与 AppTest 保留各用例边界；815 项通过不能替代真实论文质量评测。页面两次回答均有实际 Token 和工具轨迹，但未单独采样逐个增量字符，不据此给出流式延迟成绩。

## 原库、会话与运行状态

重解析前原库为 278 块/6 篇，内容指纹 `a72c99fdf44a68c1f84bcb40c43f2f4193424b43bad13b8ec32c87451e52df5d`。按最新解析得到 276 块/6 篇，指纹 `d2a023aa4dcefc76cf38a80e612210adc23b0ec4dc52a40b93fb3a6a1d380adb`；块数变化不作为丢失原文的证据，六份原文 SHA、当前块 ID 与持久化索引旁注均核对。`repair_index`仅重建向量，不更新旧解析正文，本轮从原文重新解析。

原库保留在 `data/index-before-audit-20261008`；用户已有 `data/index-backup-703095622371` 和 `.gitkeep` 删除状态未清理。原旁注备份、原会话副本与本轮独立数据均在 `.test-tmp/win-audit-deploy-20261008`，原归档区也保留。未复制其他用户问答正文到报告；会话核验仅导出数量、集合包含关系及本轮测试用户的两次问答。

原生 Streamlit 8501 与本机 Ollama 11434 保持运行。启动任务 `RAG-Audit-Streamlit-20261008` 只用于本轮启动，**无周期或开机触发器**，未承诺重启电脑后自动运行。独立容器项目 `rag-win-audit-20261008` 使用宿主回环端口 8502、单独数据目录和已有只读权重；验证结束已停止移除，镜像及测试数据保留。用户原 `diary-mysql-local` 容器仍为 exited，没有启动或删除。

## 未通过与限制

[页面样例助手初评](页面样例助手初评.json)保留两份样例，用户审核 0/2。Markdown 代号正确；ViT 回答漏答 JFT 图片数量，并将规模事实引用到第 7 页图注，原页不能支持这些规模断言。页面 100% 工具成功/任务完成只表示运行状态，不能当作语义质量。此前 F02/F14/C14/S05/S06 等残留、复杂分组表关系和正式质量审核未由本次部署覆盖；中文查询英文论文按用户要求暂缓。

本轮重解析仍报告“第 8 页三线表列边界不明确，保留原始正文与页码”，见重建日志。没有把所有表关系标为正确，也没有据单次容器探针宣称最低内存或总体性能达标。当前 Windows Docker/WSL 可用约 16GB，不改用户资源配置。

## 复测方法与异常留存

全量回归复用 `reports/模块完整性验证/verify_completeness_tests.py`；原库与缓存复用上一轮 `WindowsSSH集中复测_20261007` 的三个脚本。新增部署[指纹核验](verify_deployed_sources.py)、[原文重建](reparse_native_index.py)、[镜像构建](build_windows_image.py)、[容器执行](run_container_probe.py)、[原生状态](verify_native_state.py)均只用于本轮验证；输出存在即拒绝覆盖。容器真实模型探针复用 `独立审计修复_20261008/verify_container_runtime.py`，合成代号不计为科研样本。

初次容器探针错误地硬编码 37 个 Python 文件，实际 36 个文件全部指纹正确；检查脚本纠正后再完成种子和重启，初次失败见 `container-seed-initial-probe-error.json`。首次预检 Docker 未启动、旧 IP 失败、初次 46 项清单均原样留存。收尾首次统计把空触发器计为 1，后续只读核验排除 null，实际 0，见最终状态；服务配置没有因此改变。

所有原始日志独立保留。结论为本轮部署与工程验证通过，科研语义质量仍需修复与用户审核。
