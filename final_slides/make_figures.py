"""Figures for the final presentation (Team 10). Run: python3 make_figures.py [output_dir]

Numbers are typed in from the result files so the figures can be regenerated without the data:
  - team report (reports_docs/EXPERIMENT_FULL_REPORT.md): E1, E2, E4, E5 tables
  - final_experiments/results: rq1_models_*.csv, spec1_deltas_lgb.csv, supp5_stock_buckets.csv,
    spec2_buckets.csv, holdout_metrics.csv
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = sys.argv[1] if len(sys.argv) > 1 else 'figs'
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
INK, INK2, GRID = '#1a1a1a', '#5f5e5a', '#e4e3df'
have = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ('Arial', 'Liberation Sans', 'DejaVu Sans') if f in have), 'sans-serif')
plt.rcParams.update({
    'font.family': FONT, 'font.size': 9, 'axes.edgecolor': '#b9b8b3', 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'legend.frameon': False,
    'legend.fontsize': 8.5, 'savefig.dpi': 220, 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})


def base(ax, grid='y'):
    ax.grid(axis=grid, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def chance(ax, horizontal=True, label=True, at=0.5):
    if horizontal:
        ax.axhline(at, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        if label:
            ax.text(ax.get_xlim()[1], at, ' chance', va='center', ha='left', fontsize=8, color=INK2, clip_on=False)
    else:
        ax.axvline(at, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    print('wrote', path)


# ---------------------------------------------------------------- F1: RQ1, model families and indicator count
def fig_rq1():
    labels = ['Logistic\nregression', 'Random\nForest', 'XGBoost', 'LightGBM', 'LightGBM\n93 indicators']
    val = [0.5093, 0.5119, 0.5127, 0.5126, 0.516]
    lo = [0.5018, 0.5048, 0.5054, 0.5054, None]
    hi = [0.5170, 0.5191, 0.5201, 0.5198, None]
    fig, ax = plt.subplots(figsize=(5.7, 3.0))
    base(ax)
    ax.axhspan(0.529, 0.538, color=GREY, alpha=0.16, zorder=0, linewidth=0)
    ax.text(4.42, 0.5335, 'Kaggle leaderboard #2–#6\n(test period 2024–2026-02)', fontsize=8, color=INK2, va='center', ha='right')
    for i, v in enumerate(val):
        if lo[i] is not None:
            ax.plot([i, i], [lo[i], hi[i]], color=BLUE, linewidth=2, solid_capstyle='round', zorder=3)
            ax.plot(i, v, 'o', color=BLUE, markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=4)
        else:
            ax.plot(i, v, 'o', color='white', markersize=8, markeredgecolor=BLUE, markeredgewidth=2, zorder=4)
        ax.text(i + 0.13, v, f'{v:.3f}', va='center', ha='left', fontsize=9, color=INK, fontweight='bold')
    ax.axvline(3.5, color=GRID, linewidth=1)
    ax.text(1.5, 0.5445, '27 indicators (T27), four model families', ha='center', fontsize=8.5, color=INK2)
    ax.text(4.0, 0.5445, 'more indicators', ha='center', fontsize=8.5, color=INK2)
    ax.set_xlim(-0.5, 4.55)
    ax.set_ylim(0.495, 0.547)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Pooled AUROC, 2018–2026')
    chance(ax)
    save(fig, 'fig_rq1_models.png')


# ---------------------------------------------------------------- F2: RQ2, paired gains (team report, E1, global)
def fig_rq2_forest():
    rows = [('+ 66 technical indicators\nT27 → TALL', 0.0019, -0.0027, 0.0063, '6/9 years', GREY),
            ('+ 66 indicators on top of VIX\nT27+VIXPRE → TALL+VIXPRE', 0.0006, -0.0015, 0.0028, '4/9 years', GREY),
            ('Pre-market VIX only\nT27 → VIXPRE', 0.0529, 0.0393, 0.0661, '9/9 years', ORANGE),
            ('+ pre-market VIX\nT27 → T27+VIXPRE', 0.0589, 0.0474, 0.0702, '9/9 years', ORANGE)]
    fig, ax = plt.subplots(figsize=(5.7, 2.75))
    base(ax, 'x')
    for i, (lab, d, lo, hi, yrs, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=c, linewidth=2, solid_capstyle='round', zorder=3)
        ax.plot(d, y, 'o', color=c, markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=4)
        ax.text(hi + 0.003, y, f'{d:+.4f}   [{lo:+.4f}, {hi:+.4f}]   {yrs}', va='center', fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5, color=INK)
    ax.set_xlim(-0.012, 0.135)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0, 0.02, 0.04, 0.06])
    ax.set_xlabel('Paired ΔAUROC with 95% CI (LightGBM, global, 2018–2026)')
    save(fig, 'fig_rq2_forest.png')


# ---------------------------------------------------------------- F3: RQ2, where the gain sits (team report, E5)
def fig_rq2_attribution():
    groups = ['Official label\nclose(t) → close(t+1)', 'Overnight segment\nclose(t) → open(t+1)', 'Intraday segment\nopen(t+1) → close(t+1)']
    t27 = [0.513, 0.516, 0.507]
    vix = [0.569, 0.644, 0.506]
    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    base(ax)
    w = 0.34
    for i in range(3):
        ax.bar(i - w / 2 - 0.02, t27[i] - 0.5, w, bottom=0.5, color=BLUE, zorder=3, label='T27 (16:00)' if i == 0 else None)
        ax.bar(i + w / 2 + 0.02, vix[i] - 0.5, w, bottom=0.5, color=ORANGE, zorder=3, label='T27 + pre-market VIX (03:15)' if i == 0 else None)
        ax.text(i - w / 2 - 0.02, t27[i] + 0.003, f'{t27[i]:.3f}', ha='center', va='bottom', fontsize=8.5, color=INK)
        ax.text(i + w / 2 + 0.02, vix[i] + 0.003, f'{vix[i]:.3f}', ha='center', va='bottom', fontsize=8.5, color=INK, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(groups, fontsize=8.2)
    ax.set_ylim(0.495, 0.672)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel('AUROC (bars start at 0.5 = chance)')
    ax.legend(loc='upper left', ncol=1, handlelength=1.1, borderaxespad=0.2)
    save(fig, 'fig_rq2_attribution.png')


# ---------------------------------------------------------------- F4: RQ2, controls (spec1_deltas_lgb.csv)
def fig_rq2_controls():
    rows = [('VIX features known\nat the close', 0.0011, -0.0127, 0.0149, GREY),
            ("Yesterday's gap\n(lagged one day)", -0.0040, -0.0145, 0.0067, GREY),
            ('Gap shuffled\nacross days', -0.0019, -0.0130, 0.0087, GREY),
            ('Pre-market gap\n(03:15 ET)', 0.0633, 0.0503, 0.0757, ORANGE)]
    fig, ax = plt.subplots(figsize=(4.1, 3.0))
    base(ax, 'x')
    for i, (lab, d, lo, hi, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=c, linewidth=2, solid_capstyle='round', zorder=3)
        ax.plot(d, y, 'o', color=c, markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=4)
        ax.text(hi + 0.004, y, f'{d:+.3f}', ha='left', va='center', fontsize=8.5, color=INK, fontweight='bold' if c == ORANGE else 'normal')
    ax.axvline(0, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.2, color=INK)
    ax.set_xlim(-0.022, 0.098)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0, 0.03, 0.06])
    ax.set_xlabel('ΔAUROC vs T27 when the feature is added (95% CI)')
    save(fig, 'fig_rq2_controls.png')


# ---------------------------------------------------------------- F5: RQ4, learning strategies (team report, E1/E2)
def fig_rq4():
    modes = ['global', 'sector', 'stock', 'cross\n(unseen sector)']
    pooled = {'T27': [0.513, 0.510, 0.508, 0.514], 'VIX': [0.569, 0.566, 0.554, 0.569]}
    perday = {'T27': [0.510, 0.507, 0.505, 0.511], 'VIX': [0.511, 0.515, 0.517, 0.504]}
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 3.0), sharey=True)
    w = 0.34
    for ax, data, title in ((axes[0], pooled, 'Pooled AUROC'), (axes[1], perday, 'Per-day AUROC (same-day ranking)')):
        base(ax)
        for i in range(4):
            ax.bar(i - w / 2 - 0.02, data['T27'][i] - 0.5, w, bottom=0.5, color=BLUE, zorder=3, label='T27' if i == 0 else None)
            ax.bar(i + w / 2 + 0.02, data['VIX'][i] - 0.5, w, bottom=0.5, color=ORANGE, zorder=3, label='T27 + pre-market VIX' if i == 0 else None)
            ax.text(i + w / 2 + 0.02, max(data['VIX'][i], data['T27'][i]) + 0.002, f"{data['VIX'][i]:.3f}", ha='center', va='bottom', fontsize=8, color=INK)
        ax.set_xticks(range(4))
        ax.set_xticklabels(modes, fontsize=8.2)
        ax.set_title(title, loc='left', fontsize=8.8)
        ax.set_xlim(-0.6, 3.6)
    axes[0].set_ylim(0.495, 0.582)
    axes[0].set_ylabel('AUROC (bars start at 0.5 = chance)')
    axes[0].legend(loc='upper left', bbox_to_anchor=(0.0, -0.2), ncol=2, handlelength=1.1, borderaxespad=0)
    fig.subplots_adjust(wspace=0.08)
    save(fig, 'fig_rq4_strategies.png')


# ---------------------------------------------------------------- F6: exposure model, per-day AUROC by gap size
def fig_exposure():
    x = [0, 1, 2]
    series = [('LightGBM, one global model (29 features)', [0.512, 0.506, 0.506], ORANGE),
              ('LightGBM, one model per stock', [0.510, 0.521, 0.537], GREY),
              ('Exposure model (2 parameters)', [0.517, 0.538, 0.583], AQUA)]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    base(ax)
    for name, v, c in series:
        ax.plot(x, v, '-o', color=c, linewidth=2, markersize=7, markeredgecolor='white', markeredgewidth=1.5, zorder=3)
        ax.text(2.08, v[-1], f'{v[-1]:.3f}  {name}', va='center', ha='left', fontsize=8.5, color=INK, fontweight='bold' if c == AQUA else 'normal')
    ax.axhline(0.5, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.text(-0.27, 0.4988, 'chance', fontsize=8, color=INK2, va='top', ha='left')
    ax.set_xticks(x)
    ax.set_xticklabels(['Small VIX gap', 'Medium', 'Large VIX gap'])
    ax.set_xlim(-0.3, 4.15)
    ax.set_ylim(0.493, 0.592)
    ax.set_ylabel('Per-day AUROC (same-day stock ranking)')
    ax.spines['bottom'].set_bounds(-0.3, 2.3)
    save(fig, 'fig_exposure_buckets.png')


# ---------------------------------------------------------------- F7: RQ3, by test year (team report, E4, global)
def fig_rq3_years():
    yrs = list(range(2018, 2027))
    t27 = [0.509, 0.513, 0.536, 0.522, 0.502, 0.498, 0.513, 0.524, 0.499]
    vix = [0.603, 0.592, 0.610, 0.580, 0.537, 0.550, 0.547, 0.573, 0.525]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    base(ax)
    ax.fill_between(yrs, t27, vix, color=ORANGE, alpha=0.10, linewidth=0, zorder=1)
    for v, c, name in ((t27, BLUE, 'T27 (16:00)'), (vix, ORANGE, 'T27 + pre-market VIX (03:15)')):
        ax.plot(yrs[:-1], v[:-1], '-o', color=c, linewidth=2, markersize=6.5, markeredgecolor='white', markeredgewidth=1.4, zorder=3)
        ax.plot(yrs[-2:], v[-2:], linestyle=(0, (2, 2)), color=c, linewidth=1.6, zorder=2)
        ax.plot(yrs[-1], v[-1], 'o', color='white', markersize=6.5, markeredgecolor=c, markeredgewidth=1.8, zorder=3)
        ax.text(2026.3, v[-1] + (0.007 if c == BLUE else 0.0), name, va='center', ha='left', fontsize=8.5, color=INK)
    for y, a, b in zip(yrs, t27, vix):
        ax.text(y, 0.6285, f'+{b - a:.3f}', ha='center', va='center', fontsize=8, color=INK)
    ax.text(2026.55, 0.6285, 'gain from pre-market VIX', ha='left', va='center', fontsize=8, color=INK2)
    ax.plot([2017.6, 2026.4], [0.5, 0.5], color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xticks(yrs)
    ax.set_xticklabels([str(y) for y in yrs[:-1]] + ['2026\nJan–Feb'])
    ax.set_xlim(2017.6, 2029.3)
    ax.set_ylim(0.488, 0.636)
    ax.set_ylabel('AUROC by test year (LightGBM, global)')
    ax.spines['bottom'].set_bounds(2017.6, 2026.4)
    save(fig, 'fig_rq3_by_year.png')


# ---------------------------------------------------------------- F8: RQ3, forward test (holdout_metrics.csv)
def fig_forward():
    names = ['T27\n(16:00)', 'T27 +\npre-market VIX', 'Exposure\nmodel']
    cols = [BLUE, ORANGE, AQUA]
    pooled = [(0.5034, 0.4880, 0.5192), (0.5236, 0.5128, 0.5386), (0.5427, 0.5278, 0.5618)]
    perday = [(0.5007, 0.4848, 0.5166), (0.5038, 0.4838, 0.5238), (0.5471, 0.5243, 0.5729)]
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.0), sharey=True)
    for ax, data, title in ((axes[0], pooled, 'Pooled AUROC'), (axes[1], perday, 'Per-day AUROC')):
        base(ax)
        for i, (v, lo, hi) in enumerate(data):
            ax.plot([i, i], [lo, hi], color=cols[i], linewidth=2, solid_capstyle='round', zorder=3)
            ax.plot(i, v, 'o', color=cols[i], markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=4)
            ax.text(i + 0.14, v, f'{v:.3f}', va='center', ha='left', fontsize=9, color=INK, fontweight='bold')
        ax.axhline(0.5, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels(names, fontsize=8.2)
        ax.set_xlim(-0.5, 2.7)
        ax.set_title(title, loc='left')
    axes[0].set_ylim(0.476, 0.582)
    axes[0].set_ylabel('192 trading days after the dataset ends (95% CI)')
    fig.subplots_adjust(wspace=0.08)
    save(fig, 'fig_rq3_forward.png')


if __name__ == '__main__':
    for f in (fig_rq1, fig_rq2_forest, fig_rq2_attribution, fig_rq2_controls, fig_rq4, fig_exposure, fig_rq3_years, fig_forward):
        f()
