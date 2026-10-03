"""从真实生成实验结果绘图，不重新调用模型，不把关键项覆盖率称为准确率。"""

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rag-generation-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]


def main():
    """事实题、正文引用分开显示；不绘制没有统计依据的误差条。"""
    report = json.loads((ROOT / "reports/5_2_1 Prompt工程与生成策略/生成参数对比结果.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    font_path = Path("/System/Library/Fonts/PingFang.ttc")
    if not font_path.exists():
        font_path = Path("/System/Library/Fonts/STHeiti Light.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    labels = ["基线", "T=0", "T=0.4", "T=0.8", "p=0.6", "p=1.0", "k=10", "k=80"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, key, title, color in zip(axes,
            ("factual_key_coverage", "body_citation_rate"),
            ("关键事实覆盖率（规则检查，非答案准确率）", "必要正文引用编号覆盖率"), ("#2878b5", "#26846a")):
        values = [summary[key] for summary in report["summary"].values()]
        bars = ax.bar(labels, values, color=color, width=0.65)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=4, fontsize=10)
    figure.suptitle("Qwen2.5:7b 生成参数对比（每组 8 个有依据题 × 2 种子；基线 T=0.1 / p=0.9 / k=40）",
                   fontsize=13)
    figure.savefig(ROOT / "reports/5_2_1 Prompt工程与生成策略/生成参数质量对比.png", dpi=180)
    plt.close(figure)
    print("已从真实生成结果输出质量对比图")


if __name__ == "__main__":
    main()
