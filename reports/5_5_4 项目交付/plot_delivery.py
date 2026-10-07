"""只根据已完成的真实研究结果绘图，未知或失败结果不伪造为改善。"""
import argparse
import json
import os
from pathlib import Path
from statistics import mean
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/rag-delivery-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['chunking', 'routing', 'parallel'])
    parser.add_argument('--source', type=Path, help='本轮真实结果，省略时使用历史记录')
    parser.add_argument('--output', type=Path, help='新图表路径，禁止覆盖已有图表')
    args = parser.parse_args()
    font = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#78908B', '#397769', '#9CBFA7', '#BB9043', '#6A7EAE']
    folders = {'chunking': '5_1_2 文本分块策略', 'routing': '5_5_2 系统性能评估', 'parallel': '5_5_2 系统性能评估'}
    names = {'chunking': '五组分块检索对比_20261004', 'routing': 'Agent与固定RAG对照_20261004', 'parallel': '独立工具串行并行对照_20261004'}
    root = ROOT/'reports'/folders[args.stage]
    source = args.source or root/(names[args.stage]+'.json')
    target = args.output or root/(names[args.stage]+'.png')
    if target.exists(): raise FileExistsError('图表已存在，复测使用新路径')
    data = json.loads(source.read_text(encoding='utf-8'))
    assert data['status'] == 'completed'
    if args.stage == 'chunking':
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout='constrained')
        for ax, key, title in zip(axes, ['hit', 'mrr', 'recall'], ['Hit@K', 'MRR@K', '页级Recall@K']):
            for index, (color, (name, profile)) in enumerate(zip(colors, data['profiles'].items())):
                ax.plot([3, 5, 10], [profile['at_k'][str(k)][key] for k in (3, 5, 10)],
                        marker=['o', 's', '^', 'D', 'v'][index], linestyle=['-', '--', ':', '-.', '-'][index], color=color, label=name)
            ax.set(xticks=[3, 5, 10], ylim=(0, 1), title=title, xlabel='返回文档块数 K')
            ax.grid(alpha=.2)
            if key != 'mrr': ax.yaxis.set_major_formatter(PercentFormatter(1))
        axes[2].legend(fontsize=11, loc='lower right')
        fig.suptitle('同12篇论文与60题，M3E + RRF20 + BGE；五组实际分块', fontsize=15)
    elif args.stage == 'routing':
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
        for ax, key, title in zip(axes, ['tokens_mean_known', 'seconds_mean'], ['平均实际Token / 请求', '平均完整请求耗时 / 秒']):
            values = [data['profiles'][p][key] for p in ['agent', 'fixed_rag']]
            bars = ax.bar(['Agent统一决策', '每次固定RAG'], values, color=colors[:2], width=.5)
            ax.bar_label(bars, fmt='%.1f', padding=4);ax.set_title(title);ax.margins(y=.15);ax.grid(axis='y', alpha=.2)
        fig.suptitle('72题成对交错，共144请求；含拒答/失败，预热单列', fontsize=15)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
        for ax, category, title in zip(axes, ['time_keywords', 'two_knowledge'], ['时间 + 文本关键词', '两个独立知识库问题']):
            values = [mean(r['seconds'] for r in data['rows'] if r['category'] == category and r['profile'] == p) for p in ['serial', 'parallel']]
            bars = ax.bar(['串行调度', '并行调度'], values, color=colors[:2], width=.5)
            ax.bar_label(bars, fmt='%.2f', padding=4);ax.set(title=title, ylabel='完整Agent请求耗时 / 秒');ax.margins(y=.15);ax.grid(axis='y', alpha=.2)
        # 耗时包含失败请求，只描述本轮观测，不能把失败的快速返回解释为加速。
        failed = sum(not r['task_complete'] for r in data['rows'])
        fig.suptitle(f'两种工具组合各3对；未完成{failed}/{len(data["rows"])}，耗时均保留', fontsize=15)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180);plt.close(fig)
    print(target)


if __name__ == '__main__': main()
