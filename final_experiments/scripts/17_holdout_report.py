"""Prospective holdout, step 3 - conversion law by period, two figures, and RESULTS_HOLDOUT.md.
Needs: 15_holdout_eval.py eval, 16_related_work.py.
"""
import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
                     'axes.axisbelow': True, 'font.family': 'DejaVu Sans'})


def load_module(fname, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def conversion_by_period():
    th = load_module('10_supp_theory.py', 'supp_theory')
    panel = C.load_panel()
    tf = C.test_frame(panel)
    hold = pd.read_pickle(os.path.join(C.DATA, 'holdout', 'holdout_frame.pkl'))
    periods = [(str(yr), tf[tf['Date'].dt.year == yr], False) for yr in range(2018, 2026)] + [('2026-02 to 11 (holdout)', hold, True)]
    rows = []
    for nm, d, is_hold in periods:
        dd = d.groupby('Date').agg(g=('gap_z60', 'first'), M=('fut_ret', 'mean')).dropna()
        a_m = C._auc((dd['M'] > 0).to_numpy(dtype='float64'), -dd['g'].to_numpy(dtype='float64'))
        rho = th.static_rho(d, 'fut_ret')
        w = d.groupby('Ticker').size().reindex(rho.index)
        w = (w / w.sum()).to_numpy()
        rows.append({'period': nm, 'holdout': is_hold, 'days': int(len(dd)), 'auc_market': a_m, 'rho_mean': float((rho.to_numpy() * w).sum()),
                     'auc_stock_predicted': float(th.predicted_stock_auc(a_m, rho.to_numpy(), w)),
                     'auc_stock_observed': C._auc(d['y'].to_numpy(dtype='float64'), -d['gap_z60'].to_numpy(dtype='float64'))})
    out = pd.DataFrame(rows)
    out['obs_minus_pred'] = out['auc_stock_observed'] - out['auc_stock_predicted']
    out.to_csv(os.path.join(C.RES, 'holdout_conversion_by_period.csv'), index=False)
    wf = out[~out['holdout']]
    print('walk-forward years: corr(predicted, observed) =', round(float(np.corrcoef(wf['auc_stock_predicted'], wf['auc_stock_observed'])[0, 1]), 3),
          '| mean abs error =', round(float(wf['obs_minus_pred'].abs().mean()), 4))
    print(out.round(4).to_string(index=False))
    return out


def fig_periods():
    wf = pd.read_csv(os.path.join(C.RES, 'relwork_last_two_years.csv'))
    ho = pd.read_csv(os.path.join(C.RES, 'holdout_metrics.csv')).set_index('model')
    summ = json.load(open(os.path.join(C.RES, 'holdout_summary.json')))
    series = [('LightGBM, 27 indicators (16:00)', 'LGB T27 (16:00)', GREY, 'D'), ('LightGBM, 27 indicators + VIX gap (03:15)', 'LGB T27+VIXPRE', ORANGE, 's'),
              ('M1: rho x gap, 2 parameters (03:15)', 'M1 rho x gap', BLUE, 'o')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), sharex=True)
    xl = ['2018 to 2024\nwalk-forward\n1,761 days', '2025\nwalk-forward\n250 days', f"2026-02 to 2026-11\nfrozen models\n{summ['days']} days"]
    for ax, (v, lo, hi, hv, hlo, hhi, title, yl) in zip(axes, (('auroc', 'auroc_lo', 'auroc_hi', 'auroc_y', 'auroc_y_lo', 'auroc_y_hi', 'Pooled AUROC, official label', (0.47, 0.625)),
                                                           ('perday', 'perday_lo', 'perday_hi', 'perday', 'perday_lo', 'perday_hi', 'Per-day AUC (ranking stocks within a day)', (0.47, 0.625)))):
        for (name, key, col, mk), o, ha in zip(series, (-0.2, 0.0, 0.2), ('right', 'center', 'left')):
            a = wf[(wf.model == key) & (wf.period == '2018-2024')].iloc[0]; b = wf[(wf.model == key) & (wf.period == '2025')].iloc[0]; c = ho.loc[key]
            val = [a[v], b[v], c[hv]]; l = [a[lo], b[lo], c[hlo]]; h = [a[hi], b[hi], c[hhi]]
            ax.errorbar(np.arange(3) + o, val, yerr=[np.subtract(val, l), np.subtract(h, val)], color=col, marker=mk, ms=8, lw=0, elinewidth=2, capsize=4, label=name, zorder=3)
            for xi, vi, hh in zip(np.arange(3) + o, val, h):
                ax.annotate(f'{vi:.3f}', (xi, hh), textcoords='offset points', xytext=(0, 5), ha=ha, color=INK, fontsize=9)
        ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.axvspan(1.5, 2.5, color=GRID, alpha=0.45, zorder=0)
        ax.set_xticks(range(3), xl); ax.set_xlim(-0.5, 2.5); ax.set_ylim(*yl)
        ax.set_title(title, loc='left', fontsize=11.5, color=INK, pad=10)
    axes[1].legend(frameon=False, loc='upper left', fontsize=9.5)
    fig.text(0.01, -0.03, 'Shaded: dates after the Kaggle file ends; models trained once on data up to 2026-02-17. Bars: 95% CI, 20-day block bootstrap (10 blocks in the shaded period). Dashed line: chance.',
             color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG, 'fig_H1_unseen_period.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def fig_conversion(cv):
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    lim = (0.50, 0.66)
    ax.plot(lim, lim, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    for r in cv.itertuples():
        col = ORANGE if r.holdout else BLUE
        ax.scatter(r.auc_stock_predicted, r.auc_stock_observed, s=70, color=col, marker='s' if r.holdout else 'o', zorder=3)
        lab = f"{'2026 holdout' if r.holdout else r.period}  (rho {r.rho_mean:.2f})"
        dx, dy = (8, -4)
        if r.period in ('2019', '2023'):
            dx, dy = (8, -12)
        if r.period in ('2021',):
            dx, dy = (-86, 6)
        if r.period in ('2022',):
            dx, dy = (-98, -4)
        ax.annotate(lab, (r.auc_stock_predicted, r.auc_stock_observed), textcoords='offset points', xytext=(dx, dy), color=INK, fontsize=9)
    ax.scatter([], [], s=70, color=BLUE, marker='o', label='Walk-forward test years')
    ax.scatter([], [], s=70, color=ORANGE, marker='s', label='After the Kaggle file ends')
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel('Predicted: market-level AUROC of the VIX gap, diluted by rho')
    ax.set_ylabel('Observed stock-level AUROC of the VIX gap')
    ax.set_title('Each period\'s stock-level AUROC follows from market-level skill and rho', loc='left', fontsize=11.5, color=INK, pad=10)
    ax.legend(frameon=False, loc='upper left', fontsize=9.5)
    ax.text(0, -0.17, 'One point per period; dashed line: perfect agreement. Signal: minus the pre-market VIX gap, no fitted parameters.', transform=ax.transAxes, color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG, 'fig_H2_conversion_by_period.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def md(df, floatfmt='{:.4f}'):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: '' if pd.isna(v) else floatfmt.format(v))
    lines = ['| ' + ' | '.join(map(str, df.columns)) + ' |', '| ' + ' | '.join(['---'] * len(df.columns)) + ' |']
    lines += ['| ' + ' | '.join(map(str, r)) + ' |' for r in df.itertuples(index=False)]
    return '\n'.join(lines)


def results_md():
    r = lambda n: pd.read_csv(os.path.join(C.RES, n))
    j = lambda n: json.dumps(json.load(open(os.path.join(C.RES, n))), indent=2, ensure_ascii=False)
    out = ['# 前瞻測試與文獻對照（2026-11-22 下載並執行）', '',
           'Kaggle 資料止於 2026-02-18。之後的行情由 Yahoo Finance 另行下載（外部資料），模型只用 2026-02-17 以前的資料訓練一次，不再重新擬合。',
           '預期在下載前寫下（最後一節）。CI 為配對 20 日區塊 bootstrap 的 95% 區間；holdout 只有 10 個區塊，區間偏粗。', '',
           '## 下載與對帳', '', '```json', j('holdout_download_checks.json'), '```', '', '重建特徵與既有面板在重疊期間（2025-03 至 2026-02）的比對：', '', md(r('holdout_feature_check.csv'), '{:.6f}'), '',
           '## Holdout 結果', '', '```json', j('holdout_summary.json'), '```', '', md(r('holdout_metrics.csv')), '', md(r('holdout_deltas.csv')), '', md(r('holdout_buckets.csv')), '',
           '換算式（holdout）：', '', md(r('holdout_conversion.csv')), '', '換算式（逐期間）：', '', md(r('holdout_conversion_by_period.csv')), '',
           '## Walk-forward 的最後兩年（既有資料）', '', md(r('relwork_last_two_years.csv')), '', md(r('relwork_last_two_years_deltas.csv')), '',
           '## Krauss 等人（2017）協定，依其子期間切分（本組 LightGBM；報酬單位 %）', '', md(r('relwork_krauss_windows.csv'), '{:.3f}'), '',
           '## 下載前寫下的預期', '', '```json', j('holdout_expectations.json'), '```', '']
    open(os.path.join(C.ROOT, 'RESULTS_HOLDOUT.md'), 'w', encoding='utf-8').write('\n'.join(out))
    print('RESULTS_HOLDOUT.md written')


if __name__ == '__main__':
    cv = conversion_by_period()
    fig_periods()
    fig_conversion(cv)
    results_md()
