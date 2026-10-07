# Windows部署教程

适用于Windows x64原生部署，命令使用PowerShell，示例目录为`D:\projects\RAG+Agent`。2026-10-07已在用户实际目录`E:\工作\RAG-Agent`完成原生业务复测和隔离Docker离线运行验收；原生Python3.10.10、Ollama0.35.1、Qwen2.5:7b，容器Ollama固定0.34.0。Embedding/BGE仍用CPU，原生Qwen用RTX4060 Laptop GPU；正式性能正在按冻结评测集执行。

## 1 部署流程与机器准备

流程：复制当前代码与模型 → 安装Python → 创建Windows虚拟环境 → 安装依赖 → 启动Windows版Ollama → 准备四类模型文件 → 启动Streamlit → 反馈部署信息 → 开展Windows验收。

采用Windows 11或Windows 10 22H2及以上的64位Intel/AMD机器。Ollama的Windows和显卡支持要求见[官方Windows文档](https://docs.ollama.com/windows)及[硬件说明](https://docs.ollama.com/gpu)。本教程不适用于Windows ARM原生环境。

针对当前7B模型，建议准备至少16GB内存、30GB空闲磁盘；有支持的独立显卡时，8GB及以上显存更利于减少CPU/GPU混合加载。这是部署预估，不是本项目已测最低配置；Qwen2.5:7b的Q4_K_M权重约4.7GB，还需上下文和运行内存。[模型官方说明](https://ollama.com/library/qwen2.5:7b)。

首先保留当前`embedding.device: cpu`及`retrieval.reranker_device: cpu`，由Ollama使用受支持的GPU运行Qwen。PyTorch是否支持CUDA，不决定Ollama是否使用GPU。无受支持显卡也可使用CPU，但正式报告必须记录实际运行方式。

## 2 将当前项目复制到Windows

代码通过Git同步，先提交并推送需要部署的修复，再在Windows根目录拉取相同提交；用 `git rev-parse --short HEAD` 核对。权重和评测论文不随Git提交，需另外复制。不要覆盖已有未提交修改或原始索引。

不迁移Mac的`.venv`、`__pycache__`、`.DS_Store`及`data/models/ollama/runtime`可执行文件。Windows重新创建虚拟环境，使用Windows版Ollama。

首次Windows部署建议使用新索引和新会话，不复制Mac的`data/index`、`data/sessions`及`logs`到新环境；原Mac目录和记录保留，不删除。原始论文可复制，随后由Windows页面重新导入。正式评测还需复制Git忽略的`data/raw/evaluation_vision_transformers`下12篇论文。

已有模型可以直接复制以下完整目录，省去重新下载：

| 复制目录 | 用途 |
| --- | --- |
| `data/models/m3e-base/` | 文本与问题向量化，包含权重、配置和分词文件 |
| `data/models/bge-reranker-base/` | 候选重排序，包含权重与分词文件 |
| `data/models/qwen2.5-tokenizer/` | Agent历史Token计数的固定词表 |
| `data/models/ollama/models/` | Qwen权重，完整保留`blobs`和`manifests` |

这些模型数据可跨平台复制；不要只复制最大的权重文件，也不要迁移Mac运行程序。复制到上述相同相对路径后，通常无需修改YAML。

## 3 安装Python和依赖

从[Python3.10.10官方发布页](https://www.python.org/downloads/release/python-31010/)下载Windows installer (64-bit)，安装时选中Python Launcher及Add Python to PATH。安装完成后重新打开PowerShell。

所有命令从项目根目录运行，不从`src`目录运行。以下直接指定虚拟环境Python，不依赖PowerShell激活脚本。

```powershell
Set-Location 'D:\projects\RAG+Agent'
py -3.10 --version
# 若版本不是3.10.10，先调整Python安装/路径再继续。
py -3.10 -c "import sys; assert sys.version_info[:3] == (3, 10, 10), sys.version"
py -3.10 -m venv .venv

# 首轮保留CPU版PyTorch；Ollama的GPU推理有自己的运行库。
& .\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install tzdata
& .\.venv\Scripts\python.exe -m pip check
```

PyTorch版本和CPU安装源依据[官方旧版本安装说明](https://docs.pytorch.org/get-started/previous-versions/)。不需要额外安装torchvision、torchaudio或LangChain高层Agent框架。

项目使用`Asia/Shanghai`时区；Windows通常没有IANA时区数据库，因此显式确保`tzdata`已安装，可能已由其它依赖带入。[Python官方说明](https://docs.python.org/3.10/library/zoneinfo.html)。

## 4 启动Windows版Ollama

为保持本项目版本和启动参数一致，本教程使用固定版本的独立运行包。打开[Ollama0.34.0官方发布页](https://github.com/ollama/ollama/releases/tag/v0.34.0)，在Assets下载`ollama-windows-amd64.zip`，完整解压到：

```text
D:\projects\RAG+Agent\data\models\ollama\runtime_windows\
├── ollama.exe
└── ...运行库及其它原包文件
```

解压后必须能在这个目录直接找到`ollama.exe`，避免多套一层同名文件夹。保留原包全部文件和目录；仅复制exe可能缺GPU库。独立运行包及环境变量的使用方式见[Ollama官方Windows文档](https://docs.ollama.com/windows)。AMD设备可能需要同一版本的额外ROCm包，按官方硬件说明和具体显卡处理；不要以NVIDIA步骤推断AMD兼容性。

如果Windows已经安装Ollama桌面应用且占用11434，先从托盘正常退出该应用；本教程随后运行独立实例，避免两个服务抢同一端口。

在**终端A**执行并保持窗口开启：

```powershell
Set-Location 'D:\projects\RAG+Agent'
$env:OLLAMA_MODELS = "$PWD\data\models\ollama\models"
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_CONTEXT_LENGTH = '8192'
& .\data\models\ollama\runtime_windows\ollama.exe serve
```

以上变量只设置当前终端；每次重新启动该终端，都执行这组命令。模型地址与YAML默认`http://localhost:11434`一致，服务和页面都在同一Windows机器。

在**终端B**执行：

```powershell
Set-Location 'D:\projects\RAG+Agent'
& .\data\models\ollama\runtime_windows\ollama.exe list
# 若列表中没有qwen2.5:7b，再联网下载；已完整复制模型时无需重复下载。
& .\data\models\ollama\runtime_windows\ollama.exe pull qwen2.5:7b
```

如果模型已在列表中，可跳过pull。Ollama模型与Hugging Face Embedding/重排是不同组件，pull Qwen不会准备M3E、BGE或历史词表。

## 5 准备M3E、BGE和Qwen历史词表

若第2步已完整复制这三个目录，可以跳过下载。本节只做首次联网准备，运行时不会自动下载缺失文件。

在终端B、项目根目录一次性粘贴以下整块PowerShell代码。它读取当前YAML中的固定模型版本，并下载固定Qwen词表；没有安装另一种Embedding或替换配置。

```powershell
# 下载阶段需要联网；仅清除当前终端可能已有的离线开关。
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
$env:PYTHONUTF8 = '1'
$OutputEncoding = [System.Text.UTF8Encoding]::new()
@'
from pathlib import Path
import hashlib
from huggingface_hub import snapshot_download, hf_hub_download
from src.utils.config import load_config

# 模型名称、revision和本地目录统一读取现有配置。
config = load_config()
embedding = config["embedding"]
snapshot_download(
    repo_id=embedding["model"], revision=embedding["revision"],
    local_dir=embedding["local_path"], max_workers=2,
    allow_patterns=["*.json", "vocab.txt", "model.safetensors"],
)
retrieval = config["retrieval"]
snapshot_download(
    repo_id=retrieval["reranker_model"], revision=retrieval["reranker_revision"],
    local_dir=retrieval["reranker_local_path"], max_workers=2,
    allow_patterns=["*.json", "model.safetensors", "sentencepiece.bpe.model"],
)
# 只下载历史计数词表，不下载Hugging Face版7B生成权重。
tokenizer = hf_hub_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct", filename="tokenizer.json",
    revision="a09a35458c702b33eeacc393d103063234e8bc28",
    local_dir=str(Path(config["memory"]["tokenizer_path"]).parent),
)
assert hashlib.sha256(Path(tokenizer).read_bytes()).hexdigest() == "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
print("本地模型与历史词表准备完成")
'@ | & .\.venv\Scripts\python.exe -
```

准备代码沿用本项目手册，下载参数依据[Hugging Face0.26.5官方文档](https://huggingface.co/docs/huggingface_hub/v0.26.5/guides/download)。M3E权重约409MB，BGE约1.11GB，词表约7MB，另有配置文件。

最终目录至少包括以下关键文件及下载的完整配套文件：

```text
data/models/
├── m3e-base/model.safetensors
├── m3e-base/modules.json
├── m3e-base/1_Pooling/config.json
├── bge-reranker-base/model.safetensors
├── bge-reranker-base/sentencepiece.bpe.model
├── qwen2.5-tokenizer/tokenizer.json
└── ollama/
    ├── models/blobs/...
    ├── models/manifests/...
    └── runtime_windows/ollama.exe
```

网络访问Hugging Face受限时，可以使用第2步的已准备模型复制方案。不要把不完整的下载目录当已准备；也不要为了下载失败而更换固定revision。

## 6 启动应用

保留终端A的Ollama进程，在终端B执行：

```powershell
Set-Location 'D:\projects\RAG+Agent'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:RAG_OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
& .\.venv\Scripts\python.exe -m streamlit run src/frontend/app.py --server.address=127.0.0.1 --server.port=8501 --browser.gatherUsageStats=false
```

浏览器打开[本机应用](http://127.0.0.1:8501)。原生Windows使用回环地址；不要把仅供容器网络使用的`http://ollama:11434`填入原生配置。

部署后的操作顺序：

1. 点击“检查服务状态”，确认模型已安装。新机器索引显示“未初始化”是正常情况，不是LLM故障。
2. 在左侧上传一篇PDF，点击“开始导入”，等待解析、分块、索引完成。
3. 在RAG标签提出该篇论文的事实问题，查看流式答案、引用及原文物理页。
4. 在Agent标签询问该篇论文或计算`3.14乘以2.56`，查看工具轨迹和指标；再切换或刷新会话确认历史可读取。
5. 实际问答加载Qwen后，在另一个终端、项目根目录执行`& .\data\models\ollama\runtime_windows\ollama.exe ps`，记录PROCESSOR列。`100% GPU`、`100% CPU`、混合加载含义见[Ollama FAQ](https://docs.ollama.com/faq)。

首次加载较慢，不能把下载、冷启动或页面健康检查耗时直接当正式问答性能。GPU是否启用以实际Ollama记录为准，不以“装了显卡驱动”推断。

## 7 重启、停止和数据保存

日常使用先执行第4节终端A的服务命令，再执行第6节终端B的应用命令。通常不用再次安装依赖或下载模型。

停止时在两个终端分别按Ctrl+C。原文`data/raw`、索引`data/index`、会话`data/sessions`、模型`data/models`和`logs`均在项目目录保留。只需复制/备份相关目录，不手工编辑Chroma内部文件；备份索引或SQLite前先停止应用，避免在写入中复制。

### 7.1 已有索引出现Label not found时的停机维护

先停止所有连接本项目索引的Streamlit／Python进程，在项目根目录同步本轮全部修复后执行一次：

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe -m src.retrieval.repair_index --activate
```

程序只读取原库的块ID、正文和来源，在旁边新建索引，用当前配置的本地Embedding重新编码；构建进程退出后，另一进程核对全部正文、全部持久化向量和原生HNSW查询。两阶段成功且原库未变化才替换，原库保留为`index-backup-<编号>`，终端打印实际备份路径。失败时保留原库和失败目录，不覆盖重跑、不把临时查询恢复当修复成功；目录被占用时先核对应用已停止。完成后按第6节重启一次，集中复测四个问题。该流程已在Mac临时索引实测，Windows原库仍需执行后核验；正式部署／断网／性能验收另行安排。

## 8 常见问题

| 现象 | 处理 |
| --- | --- |
| `py`不可用或版本不对 | 重新打开终端；用`py -0p`检查Launcher识别的路径；必要时用Python3.10.10的完整exe路径创建环境 |
| `No module named src` | 返回项目根目录，使用本教程的`python.exe -m streamlit run ...`入口 |
| 安装提示需要Microsoft Visual C++ | 针对实际失败的扩展安装[Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)的C++构建工具/Windows SDK，然后重试；没有此错误无需预先安装 |
| `ZoneInfoNotFoundError: Asia/Shanghai` | 在本项目虚拟环境安装`tzdata`后重启应用 |
| 11434端口被占用 | 正常退出已有Ollama托盘实例，或明确复用已正确配置的服务；不要重复启动 |
| 模型列表为空 | 终端A的`OLLAMA_MODELS`指向项目models目录；复制必须同时包含blobs和manifests，必要时在该服务启动后pull |
| 找不到M3E/BGE/词表 | 检查第5节完整目录、固定版本及当前工作目录；重新准备后重启应用 |
| 显示CPU或混合加载 | 查看具体显卡支持、驱动和可用显存；M3E/BGE默认CPU是正常配置，Ollama单独检查PROCESSOR |
| Hugging Face下载失败 | 检查网络，或从Mac复制已经准备好的完整目录；无需切换云端推理 |

## 9 部署完成后提供的信息

告诉我Windows项目路径、系统版本、CPU、内存、显卡与显存，提供`ollama.exe --version`、`ollama.exe list`以及问答后的`ollama.exe ps`输出，说明页面能否打开、论文能否入库、回答是否返回。若失败，保留终端完整错误。

我随后根据可用的Windows访问方式开展自动回归、完整链路、持久化、边缘场景及正式性能测试；远程连接需要你提供已有的访问入口。暂停时的组合路由问题及未完成实验仍按[暂停交接说明](../reports/5_5_4%20项目交付/暂停与Windows复测说明_20261004.md)处理，不把Mac结果写成Windows验收。

原生部署完成后再处理Docker交付。当前Compose未配置GPU直通，直接启动默认是CPU方案；已有Docker步骤见[用户使用手册](用户使用手册.md)。容器结果与原生GPU结果分别记录。

## 9 2026-10-07容器验收与正式评测准备

[真实容器记录](../reports/5_5_4%20项目交付/Windows容器离线验收_20261007/README.md)包含首次失败、最终镜像指纹、CPU真实Agent、网络隔离和重启恢复。Docker Desktop4.81.0／Engine29.6.1在internal-only网络下未发布宿主页面端口；最终Compose使用固定TCP入口发布回环8501，app/ollama继续仅连internal网络。验收副本使用8502，避免占用已有原生8501。默认Compose不启用GPU，CPU功能验收耗时不能当成原生GPU性能。

先准备依赖、镜像和全部模型，再进入离线运行。桌面Ollama权重可能位于 `$env:USERPROFILE\.ollama\models`，默认Compose却挂项目 `data/models/ollama/models`；需复制完整blobs/manifests或在自己的Compose副本中显式只读挂载已有目录，原服务的权重目录不移动。Ollama的版本可按实际环境记录，不为了教程版本覆盖正在使用的服务。

正式评测使用隔离副本，保持12篇固定原文、60题、配置和源码一致，并顺序运行五组检索、两组Agent、固定RAG路由对照与串并行实验；准备资源采样依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r 'reports\5_5_2 系统性能评估\requirements-evaluation.txt'
$env:PYTHONUTF8 = '1'
```

Git默认CRLF转换会改变JSON清单与源码的原始字节哈希。冻结实验副本应保持提交原始LF文本，并先核对语料中的manifest_sha256；只做换行转换的文件须逐项确认与Git原始内容相等，有源码补丁的文件单独记录。不能通过删掉哈希检查、改变标注或覆写既有结果来使实验继续。
