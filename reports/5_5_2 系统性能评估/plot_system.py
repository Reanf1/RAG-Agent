"""根据已完成的真实评测生成PNG；未填人工评分时不绘制虚假的质量分数。"""

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rag-system-matplotlib"))
font_cache = Path(tempfile.gettempdir()) / "rag-system-font-cache"
font_cache.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    agent = json.loads(args.agent.read_text(encoding="utf-8"))
    assert retrieval["status"] == agent["status"] == "completed"
    assert retrieval["inputs"] == agent["inputs"]
    fonts = [Path("/System/Library/Fonts/PingFang.ttc"),
             Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")]
    font = next((p for p in fonts if p.exists()), None)
    if font is not None:
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.right": False, "axes.spines.top": False})
    output = args.output_dir
    targets = [output / name for name in ("检索配置与质量.png", "Agent决策与响应.png", "检索分组表现.png")]
    if any(p.exists() for p in targets):
        raise FileExistsError("图表已存在，复测请使用新输出目录")
    output.mkdir(parents=True, exist_ok=True)
    profiles = list(retrieval["profiles"].values())
    labels = [p["label"].replace("/", "\n") for p in profiles]
    colors = ["#6B7A89", "#4778A8", "#39856C", "#76A89B", "#C28739"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), layout="constrained")
    for ax, key, title, percent in zip(axes,
            ("hit_at_5", "mrr_at_5", "recall_at_5", "latency_mean_ms"),
            ("Hit@5（至少一个标注页）", "MRR@5（块排名）", "Recall@5（标注页）", "平均检索延迟 / ms"),
            (True, False, True, False)):
        values = [p[key] for p in profiles]
        bars = ax.bar(labels, values, color=colors)
        if key != "latency_mean_ms":
            ax.set_ylim(0, 1)
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.bar_label(bars, labels=[f"{v:.1%}" if percent else f"{v:.3f}" if key == "mrr_at_5" else f"{v:.0f}" for v in values], padding=3)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"12篇论文，{retrieval['question_count']}题，Top-5；各配置真实单遍运行", fontsize=14)
    fig.savefig(targets[0], dpi=180)
    plt.close(fig)

    profiles = list(agent["profiles"].values())
    labels = ["默认规则路由", "关闭规则路由"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, key, title in zip(axes.flat,
            ("tool_selection_accuracy", "iterations_mean", "latency_mean_seconds", "tokens_mean_known_requests"),
            ("工具选择准确率", "平均ReAct轮次", "平均完整响应延迟 / 秒", "平均实际Token / 用量完整请求")):
        values = [p[key] for p in profiles]
        bars = ax.bar(labels, values, color=["#4778A8", "#C28739"], width=.5)
        ax.bar_label(bars, labels=[f"{v:.1%}" if key == "tool_selection_accuracy" else f"{v:.2f}" for v in values], padding=4)
        if key == "tool_selection_accuracy":
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"同{agent['question_count']}题，两组交错；失败题保留，预热单列", fontsize=14)
    fig.savefig(targets[1], dpi=180)
    plt.close(fig)

    profiles = list(retrieval["profiles"].values())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    keys = ("fact", "comparison", "synthesis", "reasoning")
    for ax, groups, field, title in ((axes[0], keys, "by_category", "四类题目的页级Recall@5"),
                                   (axes[1], ("zh", "en"), "by_language", "中文/英文查询的页级Recall@5（原文均英文）")):
        for offset, p in enumerate(profiles):
            positions = [i + (offset - 2) * .15 for i in range(len(groups))]
            ax.bar(positions, [p[field][g]["recall_at_5"] for g in groups], width=.15,
                   label=p["label"], color=colors[offset])
        ax.set_xticks(range(len(groups)), ["事实", "对比", "归纳", "推理"] if field == "by_category" else ["中文查询", "英文查询"])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.savefig(targets[2], dpi=180)
    plt.close(fig)
    print("已从逐题真实结果生成3幅性能图表；人工评分仍待评阅人填写。")


if __name__ == "__main__":
    main()
