"""把已完成的真实检索实验绘成独立 PNG，不重新运行模型。"""

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rag-retrieval-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    """质量只使用第一测量轮；耗时使用三轮真实记录，不绘伪造误差条。"""
    report = json.loads((ROOT / "reports/检索三档对比结果.json").read_text(encoding="utf-8"))
    font_path = Path("/System/Library/Fonts/PingFang.ttc")
    if not font_path.exists():
        font_path = Path("/System/Library/Fonts/STHeiti Light.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    methods = list(report["methods"].values())
    labels = [method["label"] for method in methods]
    colors = ["#536878", "#2878b5", "#26846a"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, metric, title in zip(axes, ("hit_at_5", "recall_at_5", "mrr_at_5"), ("Hit@5", "Recall@5", "MRR@5")):
        values = [method[metric] for method in methods]
        bars = ax.bar(labels, values, color=colors, width=0.6)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        percent = metric != "mrr_at_5"
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.bar_label(bars, labels=[f"{value:.2%}" if percent else f"{value:.4f}" for value in values], padding=5)
    figure.suptitle("检索质量三档对比（6 篇论文，96 条配对查询，最终 Top-5）", fontsize=14)
    figure.savefig(ROOT / "reports/检索质量三档对比.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    groups = ("zh->zh", "en->en", "zh->en", "en->zh")
    positions = np.arange(len(groups))
    for i, method in enumerate(methods):
        axes[0].bar(positions + (i - 1) * 0.25,
                    [method["groups"][group]["hit_at_5"] for group in groups],
                    width=0.25, label=method["label"], color=colors[i])
    axes[0].set_xticks(positions, ["中文→中文", "英文→英文", "中文→英文", "英文→中文"])
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title("分组 Hit@5（每组 24 条查询）")
    axes[0].legend(frameon=False, fontsize=10)
    axes[0].grid(axis="y", alpha=0.18)
    axes[0].set_axisbelow(True)
    timings = [[elapsed for row in method["queries"] for elapsed in row["latency_ms_runs"]] for method in methods]
    boxes = axes[1].boxplot(timings, patch_artist=True, showmeans=True)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.75)
    axes[1].set_xticks([1, 2, 3], labels)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("完整检索延迟 / 毫秒（对数坐标）")
    axes[1].set_title(f"预热后延迟（每档 {report['runs']} × 96 次）")
    axes[1].grid(axis="y", alpha=0.18)
    figure.suptitle("语言差异与模型重排的耗时取舍", fontsize=14)
    figure.savefig(ROOT / "reports/检索分组与延迟.png", dpi=180)
    plt.close(figure)
    print("已生成两幅真实数据图表")


if __name__ == "__main__":
    main()
