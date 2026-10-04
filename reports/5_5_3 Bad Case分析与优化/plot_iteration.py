"""绘制真实优化前后指标，结构通过率与人工质量分数分开。"""
import json
import os
from pathlib import Path

# 使用独立字体缓存，不依赖系统Fontconfig的受限写入目录。
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/rag-bad-case-font-cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rag-bad-case-mpl-cache")
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / "优化前后对比数据.json").read_text())
    font = FontProperties(fname="/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    before, after = data["before"]["metrics"], data["after"]["metrics"]
    labels = ["优化前", "优化后"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    metrics = [
        ("工具选择路径准确率（%）", [100 * before["tool_selection_accuracy"], 100 * after["tool_selection_accuracy"]]),
        ("平均推理轮次", [before["iterations_mean"], after["iterations_mean"]]),
        ("平均响应耗时（秒）", [before["latency_mean_seconds"], after["latency_mean_seconds"]]),
        ("每题实际Token均值", [before["tokens_mean_known_requests"], after["tokens_mean_known_requests"]]),
        ("无效必填论文ID调用数", [data[k]["structural"]["invalid_paper_id_call_count"] for k in ("before", "after")]),
        ("重复调用终止题数", [len(data[k]["structural"]["repeated_calls_questions"]) for k in ("before", "after")]),
    ]
    for ax, (title, values) in zip(axes.flat, metrics):
        bars = ax.bar(labels, values, color=["#8794a5", "#298c79"], width=0.55)
        ax.set_title(title, fontproperties=font, fontsize=12)
        for label in ax.get_xticklabels():
            label.set_fontproperties(font)
        ax.set_ylim(0, max(values) * 1.2 if max(values) else 1)
        ax.bar_label(bars, labels=[f"{x:.2f}" for x in values], padding=3, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("同一60题开发集的一轮优化对照", fontproperties=font, fontsize=17)
    fig.text(0.5, 0.025, "实际本地模型单遍实测；工具路径准确率与来源保留不等于人工答案质量。", ha="center", fontproperties=font, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(HERE / "优化前后指标.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
