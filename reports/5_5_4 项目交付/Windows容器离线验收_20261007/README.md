# Windows容器离线验收（2026-10-07）

最终镜像 `rag-acceptance:0be8716` 在 Windows Docker Desktop 4.81.0、Engine 29.6.1、WSL2 上通过真实检索、BGE 重排、Qwen 生成、引用原文页、流式 Agent 和重启持久化验收。[完整终态](container-6.json)、[摘要](验收摘要.json)、[构建日志](build-5.log)、[实际依赖与源码指纹](environment-2.json)及[依赖检查](pip-check.log)保留。

## 结果与边界

隔离副本首次读取真实 ViT PDF，形成167块，导入及索引66.517秒；重复导入新增0块。该次数据来源为[container-2](container-2.json)，这轮在保存Agent事件时因AIMessage序列化失败而未整体通过。修正记录脚本后继续复用同一独立索引，最终container-6显式记录 `reused_acceptance_index=true`，没有把复用时间写成重新导入成绩。

最终真实Agent一轮完成，189.286秒、3335个实际Token、56条正文增量事件；回答指出ImageNet、ImageNet-21k、JFT-300M，引用 `vit.pdf` 第6物理页，原文页可读取。这是CPU容器的单样例功能耗时，不能替代Windows原生GPU的正式性能均值。

重启app、ollama、gateway后，独立进程读取167块和2条完整历史消息；[重启日志](container-final-restart.log)与container-6的 `new_process_reopen` 可核对。重启后的[真实页面截图](最终镜像页面检索.png)显示英文问题返回5个来源块，包含文件名、排序分数和物理页。页面经HTTP及WebSocket访问，不只依赖健康检查状态。

app和ollama只连接internal网络，两个业务运行时实际连接公网TCP均返回Network is unreachable；[Ollama出口检查](ollama-egress.log)、[容器](final-container-inspect.json)和[网络](final-network-inspect.json)保存。gateway复用同一镜像、仅将固定 `app:8501` 的字节流转发到宿主回环8502，用于此次验收避开用户原生8501；默认交付Compose使用回环8501。入口连接两个网络，业务进程不能利用入口发起任意外部请求。

这是预先准备镜像、依赖和模型后的网络隔离验收；宿主机未物理断网。原生8501页面、11434服务和原278块/6篇知识库保留。验收结束仅停止本次ragaccept20261007容器，挂载数据及日志保留，未删除Docker其他容器。

## 实際失败及修复

1. SSH会话不能访问桌面Docker凭据助手，首次[构建1](build-1.log)／[构建2](build-2.log)失败。用用户交互会话的临时计划任务完成构建，任务随后注销；没有将凭据写入文件。
2. Python3.10.10基础镜像的SQLite3.34.1不能满足Chroma，[首次业务失败](container-1.log)保留。Docker镜像单独加入pysqlite3-binary0.5.4与启动别名后，实际SQLite为3.46.1；原生业务依赖不增加该Linux专用包。[Chroma官方要求](https://docs.trychroma.com/docs/overview/troubleshooting)说明SQLite最低版本。
3. 验收脚本不能序列化LangChain AIMessage，[container-2失败日志](container-2.log)保留，改用公开 `model_dump` 后完成后续验收。
4. internal-only网络在本机Engine29.6.1未发布宿主端口，[旧Compose](compose-internal-port-failure.yml)、[旧网络](network-inspect.json)和[旧容器](container-inspect.json)保留。关闭NAT的尝试虽然能访问页面，但[container-4](container-4.json)证明仍可访问公网，故未采纳。最终保留internal业务网络，增加固定TCP入口；[Docker网络说明](https://docs.docker.com/compose/how-tos/networking/)可核对多网络含义。
5. 版本采样误查未安装的高层langchain，[采样失败](environment-failure.log)保留；项目实际使用langchain-core，修正采样名称，不额外安装高层框架。

## 复现依据

[本次Compose](compose.yml)记录隔离数据、只读模型及证明目录挂载；Ollama只读复用Windows用户预备的模型目录，默认交付路径仍为 `data/models/ollama/models`，需包含完整blobs和manifests。镜像/权重准备阶段可以联网，离线运行阶段不能pull模型。[镜像指纹](final-image-inspect.json)保存最终镜像及Ollama0.34.0的实际摘要。业务源码与当前提交按LF/CRLF换行等价逐文件核对，元数据保存镜像实际原始字节SHA，不声称其与Mac字节完全一致。

[服务日志](container-final-service.log)和所有首次失败均保留；功能通过不等于60题语义质量达标，也不代表任意机器可用同样时延。
