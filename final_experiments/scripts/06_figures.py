"""Static figures for the slides (English labels; light surface)."""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
                     'axes.axisbelow': True, 'font.family': 'DejaVu Sans'})


def fig_spec1():
    lev = pd.read_csv(os.path.join(C.RES, 'spec1_levels_lgb.csv'))
    g = lev.set_index(['config', 'label'])
    x = [0, 1, 2]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for lab, col, mk, name in (('official', BLUE, 'o', 'Official label: close(t) to close(t+1)'), ('intraday', ORANGE, 's', 'Intraday part: open(t+1) to close(t+1)')):
        v = [g.loc[(c, lab), 'value'] for c in ('C0', 'C2', 'C5')]
        lo = [g.loc[(c, lab), 'lo'] for c in ('C0', 'C2', 'C5')]
        hi = [g.loc[(c, lab), 'hi'] for c in ('C0', 'C2', 'C5')]
        ax.errorbar(x, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], color=col, marker=mk, ms=8, lw=2, capsize=4, label=name, zorder=3)
        for xi, vi in zip(x, v):
            ax.annotate(f'{vi:.3f}', (xi, vi), textcoords='offset points', xytext=(0, 16 if lab == 'official' else -24), ha='center', color=INK, fontsize=10)
    pl = [g.loc[(c, 'official'), 'value'] for c in ('C1', 'C3', 'C4')]
    ax.scatter([-0.18, -0.26, -0.34], pl, color=GREY, marker='D', s=42, zorder=3, label='Controls at 16:00: close VIX, lagged gap, shuffled gap')
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(x, ['16:00 ET, close of t\n(T27 only)', '03:15 ET on t+1\n(+ pre-market VIX gap)', '09:30 ET on t+1\n(+ stock overnight return)'])
    ax.set_xlim(-0.6, 2.45); ax.set_ylim(0.44, 0.74)
    ax.set_ylabel('Pooled AUROC, 2018 to 2026-02')
    ax.set_title('The later the prediction time, the easier the official label; the intraday part stays at chance or below', loc='left', fontsize=11.5, color=INK, pad=12)
    ax.legend(frameon=False, loc='upper left', fontsize=9.5)
    fig.text(0.01, 0.01, 'LightGBM, global, training rows from 2017. Bars: 95% CI, paired 20-day block bootstrap (10,000 draws).', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(C.FIG, 'fig_S1_prediction_time.png'), dpi=200)
    plt.close(fig)


def fig_spec2():
    bk = pd.read_csv(os.path.join(C.RES, 'spec2_buckets.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axes[0]
    order = ['|g|<0.5', '0.5<=|g|<1.5', '|g|>=1.5']
    names = (('M1_rho_x_gap', BLUE, 'o', 'M1: two-parameter model, rho x gap'), ('M3_gap_baserate', ORANGE, 's', 'M3: gap + per-stock base rate'),
             ('REF_lgb_T27+VIXPRE', AQUA, '^', 'LightGBM, T27 + VIXPRE'))
    for i, (m, col, mk, name) in enumerate(names):
        d = bk[bk.model == m].set_index('bucket').loc[order]
        xs = np.arange(3) + (i - 1) * 0.08
        ax.errorbar(xs, d['perday_auc'], yerr=[d['perday_auc'] - d['lo'], d['hi'] - d['perday_auc']], color=col, marker=mk, ms=8, lw=2, capsize=4, label=name, zorder=3)
        ax.annotate(f"{d['perday_auc'].iloc[-1]:.3f}", (xs[-1], d['perday_auc'].iloc[-1]), textcoords='offset points', xytext=(10, -4), fontsize=10, color=INK)
    days = bk[bk.model == 'M1_rho_x_gap'].set_index('bucket').loc[order, 'days']
    ax.set_xticks(range(3), [f'|gap z| < 0.5\n{days.iloc[0]} days', f'0.5 to 1.5\n{days.iloc[1]} days', f'>= 1.5\n{days.iloc[2]} days'])
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_ylabel('Per-day (cross-sectional) AUC'); ax.set_xlim(-0.4, 2.55); ax.set_ylim(0.46, 0.62)
    ax.set_title('Stock ranking improves with the size of the VIX gap', loc='left', fontsize=11.5, color=INK, pad=10)
    ax.legend(frameon=False, loc='upper left', fontsize=9.5)
    ax = axes[1]
    order2 = ['g<=-1.5', 'g<0 (VIX down)', 'g>0 (VIX up)', 'g>=+1.5']
    d = bk[bk.model == 'RHO_only'].set_index('bucket').loc[order2]
    xs = np.arange(4)
    ax.errorbar(xs, d['perday_auc'], yerr=[d['perday_auc'] - d['lo'], d['hi'] - d['perday_auc']], color=BLUE, marker='o', ms=8, lw=0, elinewidth=2, capsize=4, zorder=3)
    for xi, vi in zip(xs, d['perday_auc']):
        ax.annotate(f'{vi:.3f}', (xi, vi), textcoords='offset points', xytext=(10, -4), fontsize=10, color=INK)
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(xs, [f'z <= -1.5\n{int(d.days.iloc[0])} days', f'gap < 0\n{int(d.days.iloc[1])} days', f'gap > 0\n{int(d.days.iloc[2])} days', f'z >= +1.5\n{int(d.days.iloc[3])} days'])
    ax.set_xlim(-0.5, 3.7); ax.set_ylim(0.38, 0.68)
    ax.set_ylabel('Per-day AUC of rho alone'); ax.set_xlabel('Pre-market VIX gap (up = market fell overnight)')
    ax.set_title('Ranking by market correlation flips sign with the gap', loc='left', fontsize=11.5, color=INK, pad=10)
    fig.text(0.01, 0.01, 'rho: 252-day correlation of a stock with the leave-one-out market return. Official label, 2018 to 2026-02. Bars: 95% CI, block bootstrap.', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(C.FIG, 'fig_S2_rho_gap.png'), dpi=200)
    plt.close(fig)


def fig_vix_open():
    raw = C.load_panel()[['Date', 'on_ret', 'gap_next_unmasked']]
    day = raw.groupby('Date').agg(on=('on_ret', 'mean'), gap=('gap_next_unmasked', 'first')).dropna()
    rc = day['on'].rank().rolling(60).corr(day['gap'].rank())          # rolling rank correlation, robust to single days
    rc = rc['2014-07-01':'2018-12-31']
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(rc.index, rc.values, color=BLUE, lw=2)
    for dt, txt, yy, ha, dx in ((pd.Timestamp('2016-04-15'), 'Cboe starts overnight VIX\ndissemination (2016-04-15)', -0.22, 'right', -6), (pd.Timestamp('2017-01-01'), 'Mask used in the\nexperiments (2017-01-01)', -0.86, 'left', 6)):
        ax.axvline(dt, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.annotate(txt, (dt, yy), textcoords='offset points', xytext=(dx, 0), fontsize=9, color=INK2, va='top', ha=ha)
    ax.set_ylim(-1.0, -0.15)
    ax.set_ylabel('Rolling 60-day rank correlation')
    ax.set_title('The daily VIX OPEN decouples from the stock overnight move from spring 2016', loc='left', fontsize=11.5, color=INK, pad=10)
    fig.text(0.01, 0.01, 'Correlation between OPEN(t+1)/CLOSE(t)-1 of VIX (Cboe daily file) and the mean stock return close(t) to open(t+1).', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(C.FIG, 'fig_S4_vix_open.png'), dpi=200)
    plt.close(fig)
    out = {}
    for nm, a, b in (('2014-01-01..2016-04-14', '2014-01-01', '2016-04-14'), ('2016-04-15..2016-12-31', '2016-04-15', '2016-12-31'), ('2017-01-01..2018-12-31', '2017-01-01', '2018-12-31')):
        d = day[a:b]
        out[nm] = {'spearman': round(float(d['on'].corr(d['gap'], method='spearman')), 3), 'pearson': round(float(d['on'].corr(d['gap'])), 3), 'days': len(d)}
    pd.DataFrame(out).T.to_csv(os.path.join(C.RES, 'spec4_vix_open_periods.csv'))
    print(pd.DataFrame(out).T.to_string())


if __name__ == '__main__':
    os.makedirs(C.FIG, exist_ok=True)
    fig_spec1(); fig_spec2(); fig_vix_open()
    print(os.listdir(C.FIG))
