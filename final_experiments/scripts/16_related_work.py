"""Numbers that can be set beside the related work, from predictions already saved.
(1) Krauss, Do & Huck (2017) protocol on our universe, cut into their sub-periods: 31 lagged returns, beat-the-median label, long top-k / short flop-k,
    daily return before costs and after their cost assumption (0.05% per half-turn = 0.20% a day for a long-short book), annualised by compounding.
    Model: LightGBM walk-forward from 11_supp_robust.py era (their closest counterpart is the gradient-boosted tree).
(2) The last two walk-forward test years (2025 and the 31 days of 2026) as a block, for the unseen-year question.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

COST = 0.0020


def ci(d):
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def krauss_windows():
    panel = C.load_panel()
    path = os.path.join(C.PRED, 'supp8_era_preds.npy')
    if not os.path.exists(path):
        sys.exit('run 11_supp_robust.py era first')
    pk = np.load(path)[0]
    keep = (panel['Date'] >= '2005-01-01').to_numpy()
    tfe = panel.loc[keep, ['Ticker', 'Date', 'fut_ret']].reset_index(drop=True)
    tfe['y_cs'] = (tfe['fut_ret'] > tfe.groupby('Date')['fut_ret'].transform('median')).astype('float64')
    tfe['p'] = pk[keep]
    boot = C.Boot(tfe, n_boot=5000)
    days = pd.DatetimeIndex(boot.days)
    da = pd.Series(C.day_auc(tfe, tfe['p'].to_numpy(), 'y_cs')).reindex(days).to_numpy(dtype='float64')
    n_day = tfe.dropna(subset=['p']).groupby('Date').size().reindex(days).to_numpy(dtype='float64')
    wins = [('2005-01 to 2008-08', '2005-01-01', '2008-08-31'), ('2008-09 to 2009-12', '2008-09-01', '2009-12-31'), ('2010-01 to 2015-10', '2010-01-01', '2015-10-31'),
            ('2015-11 to 2017-12', '2015-11-01', '2017-12-31'), ('2018-01 to 2026-02', '2018-01-01', '2026-12-31')]
    out = []
    for k in (10, 2):
        d = tfe.dropna(subset=['p']).copy()
        d = d[d.groupby('Date')['p'].transform('size') >= 2 * k]
        d['hi'] = d.groupby('Date')['p'].rank(method='first', ascending=False); d['lo'] = d.groupby('Date')['p'].rank(method='first', ascending=True)
        top, flop = d[d['hi'] <= k], d[d['lo'] <= k]
        hit = (top.groupby('Date')['y_cs'].sum().reindex(days).fillna(0) + (1 - flop['y_cs']).groupby(flop['Date']).sum().reindex(days).fillna(0)).to_numpy(dtype=np.float32)
        cnt = (top.groupby('Date').size().reindex(days).fillna(0) + flop.groupby('Date').size().reindex(days).fillna(0)).to_numpy(dtype=np.float32)
        ls = (top.groupby('Date')['fut_ret'].mean() - flop.groupby('Date')['fut_ret'].mean()).reindex(days)
        for nm, a, b in wins:
            dm = np.asarray((days >= a) & (days <= b)) & ls.notna().to_numpy()
            r = ls.to_numpy(dtype='float64')[dm]
            w = dm.astype(np.float32)
            lsv = np.where(dm, ls.fillna(0).to_numpy(dtype='float64'), 0).astype(np.float32)
            lg0 = np.where(dm, np.log1p(ls.fillna(0).to_numpy(dtype='float64')), 0).astype(np.float32)
            lg1 = np.where(dm, np.log1p(ls.fillna(0).to_numpy(dtype='float64') - COST), 0).astype(np.float32)
            den = np.maximum(boot.W @ w, 1e-9)
            row = {'k': k, 'window': nm, 'days': int(dm.sum()), 'stocks_per_day': float(np.nanmean(n_day[dm])),
                   'accuracy': float(hit[dm].sum() / cnt[dm].sum()), 'mean_daily_ls_pct': float(100 * r.mean()),
                   'ann_before_cost_pct': float(100 * (np.exp(250 * np.log1p(r).mean()) - 1)), 'ann_after_cost_pct': float(100 * (np.exp(250 * np.log1p(r - COST).mean()) - 1))}
            row['acc_lo'], row['acc_hi'] = ci((boot.W @ (hit * dm)) / np.maximum(boot.W @ (cnt * dm), 1e-9))
            lo, hi = ci((boot.W @ lsv) / den); row['daily_lo_pct'], row['daily_hi_pct'] = 100 * lo, 100 * hi
            lo, hi = ci(np.exp(250 * (boot.W @ lg0) / den) - 1); row['ann_before_lo_pct'], row['ann_before_hi_pct'] = 100 * lo, 100 * hi
            lo, hi = ci(np.exp(250 * (boot.W @ lg1) / den) - 1); row['ann_after_lo_pct'], row['ann_after_hi_pct'] = 100 * lo, 100 * hi
            if k == 10:
                v = da[np.asarray((days >= a) & (days <= b))]
                row['perday_auc'] = float(np.nanmean(v))
            out.append(row)
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(C.RES, 'relwork_krauss_windows.csv'), index=False)
    pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
    print(out.round(3).to_string(index=False))


def last_two_years():
    panel = C.load_panel()
    tf = C.test_frame(panel)
    boot = C.Boot(tf, n_boot=10000)
    y = tf['y'].to_numpy(dtype='float64')
    rows, deltas = [], []
    models = {'LGB T27 (16:00)': 'lgb_T27', 'LGB T27+VIXPRE': 'lgb_T27+VIXPRE', 'LGB T27+VIXPRE, 2017+': 'lgb_T27+VIXPRE_common', 'M1 rho x gap': 'supp6_D0_rho_gap',
              'D1 rho x gap + selection': 'supp6_D1_rho_gap+sel'}
    for per, a, b in (('2025', '2025-01-01', '2025-12-31'), ('2025 + 2026 (31 days)', '2025-01-01', '2026-12-31'), ('2018-2024', '2018-01-01', '2024-12-31')):
        rm = ((tf['Date'] >= a) & (tf['Date'] <= b)).to_numpy()
        dm = np.asarray((pd.DatetimeIndex(boot.days) >= a) & (pd.DatetimeIndex(boot.days) <= b))
        cache = {}
        for nm, key in models.items():
            p = C.load_pred(key)[0]
            pt, dr = boot.pooled(np.where(rm, y, np.nan), p)
            v = pd.Series(C.day_auc(tf, p)).reindex(pd.DatetimeIndex(boot.days)).to_numpy(dtype='float64')
            ok = (~np.isnan(v)) & dm
            den = boot.W @ ok.astype(np.float32)
            dd = np.where(den > 0, (boot.W @ np.where(ok, v, 0).astype(np.float32)) / np.maximum(den, 1e-9), np.nan)
            cache[nm] = (pt, dr, float(v[ok].mean()), dd)
            rows.append({'period': per, 'days': int(dm.sum()), 'model': nm, 'auroc': pt, 'auroc_lo': ci(dr)[0], 'auroc_hi': ci(dr)[1], 'perday': float(v[ok].mean()),
                         'perday_lo': ci(dd)[0], 'perday_hi': ci(dd)[1]})
        for a_, b_, met in (('LGB T27 (16:00)', 'LGB T27+VIXPRE', 'pooled'), ('LGB T27+VIXPRE, 2017+', 'M1 rho x gap', 'pooled'), ('LGB T27+VIXPRE, 2017+', 'M1 rho x gap', 'perday')):
            i = (0, 1) if met == 'pooled' else (2, 3)
            A, B = cache[a_], cache[b_]
            ok = ~(np.isnan(A[i[1]]) | np.isnan(B[i[1]]))
            deltas.append({'period': per, 'a': a_, 'b': b_, 'metric': met, **C.delta_stats(A[i[1]][ok], B[i[1]][ok], A[i[0]], B[i[0]])})
    rows, deltas = pd.DataFrame(rows), pd.DataFrame(deltas)
    rows.to_csv(os.path.join(C.RES, 'relwork_last_two_years.csv'), index=False)
    deltas.to_csv(os.path.join(C.RES, 'relwork_last_two_years_deltas.csv'), index=False)
    print(rows.round(4).to_string(index=False))
    print(deltas.round(4).to_string(index=False))


if __name__ == '__main__':
    krauss_windows()
    last_two_years()
