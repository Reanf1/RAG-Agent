# 5.5.1 评测集构建

研究方向为 **Transformer 在图像分类与目标检测中的应用**。分类侧覆盖模型结构、训练数据、蒸馏与效率；检测侧沿 DETR 系列覆盖集合预测、查询设计、收敛与去噪训练。限定两个相邻任务，便于本科生阅读、对比和答辩，不扩展为整个计算机视觉领域。

## 论文来源与版本

共收集12篇真实论文、179个PDF物理页。下表链接均为作者预印本或会议正式论文入口；实际下载URL、文件大小、SHA-256、版本及本地路径详见[论文清单](论文清单.json)。年份为会议发表年份，可能不同于预印本首次发布年份。DINO指目标检测论文，不是同名自监督学习论文。

| 论文ID | 原始来源 | 年份 / 会议 | 固定PDF版本 | 物理页数 |
| --- | --- | --- | --- | ---: |
| `vit` | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929v2) | 2021 / ICLR | arXiv:2010.11929v2 | 22 |
| `deit` | [Training data-efficient image transformers & distillation through attention](https://proceedings.mlr.press/v139/touvron21a.html) | 2021 / ICML | PMLR 139:10347–10357 | 11 |
| `swin` | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html) | 2021 / ICCV | CVF ICCV 2021 accepted version | 11 |
| `pvt` | [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Pyramid_Vision_Transformer_A_Versatile_Backbone_for_Dense_Prediction_Without_ICCV_2021_paper.html) | 2021 / ICCV | CVF ICCV 2021 accepted version | 11 |
| `t2t_vit` | [Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Tokens-to-Token_ViT_Training_Vision_Transformers_From_Scratch_on_ImageNet_ICCV_2021_paper.html) | 2021 / ICCV | CVF ICCV 2021 accepted version | 10 |
| `cait` | [Going Deeper with Image Transformers](https://openaccess.thecvf.com/content/ICCV2021/html/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.html) | 2021 / ICCV | CVF ICCV 2021 accepted version | 11 |
| `detr` | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872v3) | 2020 / ECCV | arXiv:2005.12872v3 | 26 |
| `deformable_detr` | [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159v4) | 2021 / ICLR | arXiv:2010.04159v4 | 16 |
| `conditional_detr` | [Conditional DETR for Fast Training Convergence](https://openaccess.thecvf.com/content/ICCV2021/html/Meng_Conditional_DETR_for_Fast_Training_Convergence_ICCV_2021_paper.html) | 2021 / ICCV | CVF ICCV 2021 accepted version | 10 |
| `dab_detr` | [DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR](https://arxiv.org/abs/2201.12329v4) | 2022 / ICLR | arXiv:2201.12329v4 | 19 |
| `dn_detr` | [DN-DETR: Accelerate DETR Training by Introducing Query DeNoising](https://openaccess.thecvf.com/content/CVPR2022/html/Li_DN-DETR_Accelerate_DETR_Training_by_Introducing_Query_DeNoising_CVPR_2022_paper.html) | 2022 / CVPR | CVF CVPR 2022 accepted version | 9 |
| `dino` | [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605v4) | 2023 / ICLR | arXiv:2203.03605v4 | 23 |

原文位于项目的 `data/raw/evaluation_vision_transformers/`，各文件以论文ID命名，例如 `vit.pdf`。论文和衍生语料遵循现有 `.gitignore`，没有提交为代码文件；下载使用公开原始来源，不将问答文件当成论文入库。

DAB-DETR的固定版本URL下载返回406，实际通过官方未带版本的PDF入口获得v4，并核对首屏版本标记和文件哈希。该入口将来可能更新，复现程序必须通过当前哈希才能接受文件。DINO的ICLR 2023发表信息也由[作者官方仓库](https://github.com/IDEA-Research/DINO)确认。

## 问答组成

主文件维持用户指定路径：[reports/评测集.json](../评测集.json)。共60题，问题不是中英翻译后重复计数。

| 类型 | 中文 | 英文 | 合计 | 检查重点 |
| --- | ---: | ---: | ---: | --- |
| 事实性 `fact` | 8 | 7 | 15 | 数据集、模型结构、训练设置及结果 |
| 对比性 `comparison` | 7 | 8 | 15 | 两篇及以上论文的方法差异和实验条件 |
| 归纳性 `synthesis` | 8 | 7 | 15 | 跨论文归纳技术路线、指标及局限 |
| 推理性 `reasoning` | 7 | 8 | 15 | 作者动机、观察及有依据的限定推断 |
| 合计 | 30 | 30 | 60 | 四类均覆盖 |

每题包含稳定 `id`、`category`、`language`、`question`、`reference_answer`、`answer_points`、`answer_basis`、`paper_ids` 和 `evidence`。评分要点按内容判断，不要求答案逐字等于参考答案；问题语言是指定回答语言。参考答案为原文概括，推理题区分作者解释与推断，不将模型推测写成论文事实。部分评分要点使用英文术语，评分可接受正确的中文表达。

43个可复用证据位置保存在论文清单的 `evidence` 中，含短定位锚点和该段支持的内容；每题引用这些位置。`page_number` 从1开始，表示本地PDF的物理页，不是会议页码或正文印刷页码。跨论文题必须核对全部列出的来源，不能只找到一篇就认定答案完整。

## 核验与后续复现

直接复用模块一的PyMuPDF加载器和递归切分默认参数512/64字符，生成1694个候选块；候选语料 `data/raw/evaluation_vision_transformers/corpus.json` 只含12篇论文正文及来源元数据，不含问题、参考答案或评分要点。版面元数据不在每块中重复存储，可按原始PDF重新读取。

从项目根目录运行，输出必须使用新的文件名，避免覆盖历史证据：

```bash
# 原文已在本机：核验JSON、原文哈希、证据物理页及加载/分块后的锚点。
.venv/bin/python "reports/5_5_1 评测集构建/verify_dataset.py" --output "reports/5_5_1 评测集构建/评测集核验_新时间.json"

# 新环境：显式下载清单中的缺失原文，哈希一致后才接受。
.venv/bin/python "reports/5_5_1 评测集构建/verify_dataset.py" --download --output "reports/5_5_1 评测集构建/评测集下载核验_新时间.json"
```

定稿核验见[结果JSON](评测集交付核验_20261003.json)和[原始日志](评测集交付核验_20261003.log)。结构、12篇哈希、43个证据锚点及真实分块定位通过。首次未使用证据断言失败和修正后的中间结果保留，不覆盖为成功记录。详细解释见[构建QA](../../docs/QA/5.5.1%20评测集构建.md)。

## 使用范围

- 本轮仅构建并锁定评测输入，没有运行待评测模型，没有新Hit@5/MRR、Agent性能或答案分数。
- 参考答案由助手依据原文整理，未经独立人工复核；后续评分前应由学生或教师复核答案及量表。
- 12篇来源文档全部为英文，30条中文查询覆盖中文问英文论文，不代表中文原文检索已被覆盖。
- ViT原文与早期开发集存在重叠，故本集不能称为完全独立来源的盲测；其余11篇扩大研究方向覆盖。后续不能按检索排名或生成答案修改题目、答案和证据来提高分数。
- 当前相关性依据是论文与物理页。核验输出的锚点所在块仅用于定位，不等于完整的相关chunk金标准；后续计算检索指标时需明确页级匹配规则、跨页块规则及多来源覆盖，不直接沿用旧开发集的块级评分。
