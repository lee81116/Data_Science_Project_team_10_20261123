"""Figures for the supplementary experiments (English labels; light surface; same palette as 06_figures.py)."""
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
BUCKETS = ['|g|<0.5', '0.5<=|g|<1.5', '|g|>=1.5']


def read(name):
    path = os.path.join(C.RES, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{name} missing: run the supplementary scripts first')
    return pd.read_csv(path)


def fig_gap_buckets():
    s5, s6 = read('supp5_stock_buckets.csv'), read('supp6_buckets.csv')
    series = [('One global LightGBM: T27 + VIX gap', s6[s6.model == 'REF LGB T27+VIXPRE'], GREY, 'D'),
              ('100 per-stock LightGBMs: VIX gap only', s5[s5.model == 'stock_lgb_VIXPRE'], ORANGE, 's'),
              ('M1: rho x gap, 2 parameters', s6[s6.model == 'D0_rho_gap'], BLUE, 'o'),
              ('D1: M1 + selection score, 3 parameters', s6[s6.model == 'D1_rho_gap+sel'], AQUA, '^')]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    off = [-0.24, -0.08, 0.08, 0.24]
    days = s6[s6.model == 'D0_rho_gap'].set_index('bucket')['days']
    for (name, d, col, mk), o in zip(series, off):
        d = d.set_index('bucket').loc[BUCKETS]
        x = np.arange(3) + o
        ax.errorbar(x, d['perday_auc'], yerr=[d['perday_auc'] - d['lo'], d['hi'] - d['perday_auc']], color=col, marker=mk, ms=8, lw=0, elinewidth=2,
                    capsize=4, label=name, zorder=3)
        if col in (BLUE, GREY):
            for xi, vi, hi in zip(x, d['perday_auc'], d['hi']):
                ax.annotate(f'{vi:.3f}', (xi, hi), textcoords='offset points', xytext=(0, 6), ha='center', color=INK, fontsize=9.5)
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(range(3), [f'Small gap, |z| < 0.5\n{days[BUCKETS[0]]:,} days', f'Medium gap, 0.5 to 1.5\n{days[BUCKETS[1]]:,} days', f'Large gap, |z| >= 1.5\n{days[BUCKETS[2]]:,} days'])
    ax.set_xlim(-0.55, 2.55); ax.set_ylim(0.48, 0.625)
    ax.set_ylabel('Per-day AUC (ranking stocks within a day)')
    ax.set_title('Two shared parameters rank stocks better than one LightGBM or 100 per-stock models', loc='left', fontsize=11.5, color=INK, pad=12)
    ax.legend(frameon=False, loc='upper left', fontsize=9.5)
    ax.text(0, -0.2, 'Test period 2018 to 2026-02. Bars: 95% CI, 20-day block bootstrap. Dashed line: chance.', transform=ax.transAxes, color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG, 'fig_S5_perday_by_gap.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def fig_theory():
    cv, ip = read('supp7_conversion.csv'), read('supp7_implied_perday.csv')
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.9), gridspec_kw={'width_ratios': [1.15, 1]})
    cols = {'close-to-close': BLUE, 'overnight': ORANGE, 'intraday': AQUA}
    ax.plot([0.47, 0.85], [0.47, 0.85], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    for lab, col in cols.items():
        for kind, mk, face in (('signal', 'o', col), ('placebo', 'D', SURF), ('oracle', 's', col)):
            d = cv[(cv.label == lab) & (cv.signal.str.contains('placebo') if kind == 'placebo' else cv.signal.str.startswith('oracle') if kind == 'oracle'
                                         else ~(cv.signal.str.contains('placebo') | cv.signal.str.startswith('oracle')))]
            if d.empty:
                continue
            ax.errorbar(d['auc_stock_predicted'], d['auc_stock_observed'], yerr=[d['auc_stock_observed'] - d['obs_lo'], d['obs_hi'] - d['auc_stock_observed']],
                        color=col, marker=mk, mfc=face, mec=col, mew=1.6, ms=8, lw=0, elinewidth=1.6, capsize=3, zorder=3)
    for lab, col in cols.items():
        ax.plot([], [], color=col, marker='o', lw=0, ms=8, label=f'Label: {lab}')
    ax.plot([], [], color=INK2, marker='D', mfc=SURF, lw=0, ms=7, label='Placebo signals')
    ax.plot([], [], color=INK2, marker='s', lw=0, ms=7, label='Oracle: realised market return')

    def note(sig, lab, text, dx, dy):
        r = cv[(cv.signal == sig) & (cv.label == lab)].iloc[0]
        ax.annotate(text, (r['auc_stock_predicted'], r['auc_stock_observed']), textcoords='offset points', xytext=(dx, dy), color=INK, fontsize=9,
                    arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    note('-VIX gap (03:15)', 'close-to-close', 'VIX gap, official label', -128, 26)
    note('-VIX gap (03:15)', 'overnight', 'VIX gap, overnight', -108, 12)
    note('oracle: realised market return', 'overnight', 'overnight oracle', -100, 2)
    ax.set_xlim(0.47, 0.85); ax.set_ylim(0.47, 0.85)
    ax.set_xlabel('Stock-level AUROC predicted from the market-level AUROC and rho')
    ax.set_ylabel('Stock-level AUROC observed')
    ax.set_title('Conversion law: market-level skill diluted by rho', loc='left', fontsize=11.5, color=INK, pad=10)
    ax.legend(frameon=False, loc='lower right', fontsize=9)

    d = ip[ip.direction.str.startswith('direction')].set_index('bucket').loc[['all days'] + BUCKETS]
    x = np.arange(4)
    for key, lo, hi, col, mk, name, o in (('observed', 'obs_lo', 'obs_hi', BLUE, 'o', 'Observed', -0.1), ('implied', 'imp_lo', 'imp_hi', ORANGE, 's', 'Implied by the one-factor model', 0.1)):
        bx.errorbar(x + o, d[key], yerr=[d[key] - d[lo], d[hi] - d[key]], color=col, marker=mk, ms=8, lw=0, elinewidth=2, capsize=4, label=name, zorder=3)
        for xi, vi, h in zip(x + o, d[key], d[hi]):
            bx.annotate(f'{vi:.3f}', (xi, h), textcoords='offset points', xytext=(-14 if o < 0 else 14, 4), ha='center', color=INK, fontsize=9)
    bx.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    bx.set_xticks(x, ['All days', 'Small gap\n|z| < 0.5', 'Medium gap\n0.5 to 1.5', 'Large gap\n|z| >= 1.5'])
    bx.set_xlim(-0.5, 3.5); bx.set_ylim(0.49, 0.63)
    bx.set_ylabel('Per-day AUC of the rho ranking')
    bx.set_title('Cross-sectional ranking: observed vs implied', loc='left', fontsize=11.5, color=INK, pad=10)
    bx.legend(frameon=False, loc='upper left', fontsize=9.5)
    fig.text(0.01, -0.02, 'Test period 2018 to 2026-02. Bars: 95% CI, 20-day block bootstrap. Dashed lines: perfect agreement (left), chance (right).', color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG, 'fig_S7_theory_check.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def fig_era():
    e = read('supp8_era.csv')
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(e))
    series = [('Lagged returns (Krauss features), per-day AUC', 'krauss_cs_perday', 'k_lo', 'k_hi', BLUE, 'o', -0.16),
              ('27 technical indicators, pooled AUROC', 'T27_auroc', 'ta_lo', 'ta_hi', ORANGE, 's', 0.0),
              ('Zero-parameter market reversal, pooled AUROC', 'mkt_reversal_auroc', 'rev_lo', 'rev_hi', AQUA, '^', 0.16)]
    for name, v, lo, hi, col, mk, o in series:
        ax.errorbar(x + o, e[v], yerr=[e[v] - e[lo], e[hi] - e[v]], color=col, marker=mk, ms=8, lw=0, elinewidth=2, capsize=4, label=name, zorder=3)
        for xi, vi, h in zip(x + o, e[v], e[hi]):
            ax.annotate(f'{vi:.3f}', (xi, h), textcoords='offset points', xytext=(0, 5), ha='center', color=INK, fontsize=9)
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(x, [f"{r.era}\n{r.days:,} days\nabout {r.stocks_per_day:.0f} stocks a day" for r in e.itertuples()])
    ax.set_xlim(-0.5, len(e) - 0.5); ax.set_ylim(0.49, 0.555)
    ax.set_ylabel('AUC, walk-forward out of sample')
    ax.set_title('With 16:00 information the ceiling has stayed near 0.51 to 0.52 for two decades', loc='left', fontsize=11.5, color=INK, pad=12)
    ax.legend(frameon=False, loc='upper right', fontsize=9.5)
    ax.text(0, -0.27, 'Bars: 95% CI, 20-day block bootstrap. Universe: current NASDAQ-100 members (survivors only). Dashed line: chance.', transform=ax.transAxes, color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG, 'fig_S8_era.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(C.FIG, exist_ok=True)
    fig_gap_buckets()
    fig_theory()
    fig_era()
    print('figures written to', C.FIG)
