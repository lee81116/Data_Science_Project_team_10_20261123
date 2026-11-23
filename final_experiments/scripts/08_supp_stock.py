"""Supp S5 - per-stock mode rebuilt on the same panel, so global and stock come from one pipeline.
Protocol (EXPERIMENT_SETUP_REPORT.md): one model per ticker, own history only; LightGBM 4 leaves / 100 trees / min_child 40; LR C=0.01.
Usage: python 08_supp_stock.py fit <lgb|lr>   |   python 08_supp_stock.py eval
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

STOCK_LGB = dict(C.LGB_PARAMS, num_leaves=4, n_estimators=100, min_child_samples=40, n_jobs=1)
CONFIGS = {'T27': (C.T27, None), 'VIXPRE': (C.VIXPRE, C.VIXPRE), 'T27+VIXPRE': (C.T27 + C.VIXPRE, None)}
MIN_TRAIN = 120          # fewer usable training rows than this -> no prediction for that ticker-year
REPORTED = {'T27': dict(auroc=0.508, perday=0.505, on=0.509, id=0.504), 'VIXPRE': dict(auroc=0.561, perday=0.517, on=0.624, id=0.506),
            'T27+VIXPRE': dict(auroc=0.554, perday=0.517, on=0.611, id=0.504)}


def one_ticker(sub, feats, require, model, purge_days):
    """sub: one ticker's rows (sorted by date). Returns (index, p, thr) for its test rows."""
    x = sub[feats].to_numpy(dtype='float64')
    y = sub['y'].to_numpy(dtype='float64')
    d = sub['Date']
    ok = ~np.isnan(y)
    if require is not None:
        ok &= sub[require].notna().all(axis=1).to_numpy()
    p = np.full(len(sub), np.nan); thr = np.full(len(sub), np.nan)
    for yr in C.TEST_YEARS:
        te = (d.dt.year == yr).to_numpy()
        if not te.any():
            continue
        tr = ((d < pd.Timestamp(yr, 1, 1)) & (d != purge_days[yr])).to_numpy() & ok
        if tr.sum() < MIN_TRAIN or len(np.unique(y[tr])) < 2:
            continue
        try:
            p[te] = C.fit_predict(x[tr], y[tr], x[te], model, lgb_params=STOCK_LGB)
            thr[te] = y[tr].mean()
        except Exception as e:                                   # keep going; the ticker-year stays NaN
            print(f'  fit failed {sub["Ticker"].iloc[0]} {yr}: {e}', flush=True)
    return sub.index.to_numpy(), p, thr


def fit(model):
    panel = C.load_panel()
    dates = pd.DatetimeIndex(np.sort(panel['Date'].unique()))
    purge_days = {yr: dates[dates < pd.Timestamp(yr, 1, 1)].max() for yr in C.TEST_YEARS}   # same purge day as the global protocol
    is_test = (panel['Date'] >= '2018-01-01').to_numpy()
    groups = [g for _, g in panel.groupby('Ticker', sort=True)]
    for name, (feats, require) in CONFIGS.items():
        res = Parallel(n_jobs=4)(delayed(one_ticker)(g[['Ticker', 'Date', 'y'] + feats], feats, require, model, purge_days) for g in groups)
        p = np.full(len(panel), np.nan); thr = np.full(len(panel), np.nan)
        for idx, pi, ti in res:
            p[idx] = pi; thr[idx] = ti
        C.save_pred(f'stock_{model}_{name}', p[is_test], thr[is_test])
        print(model, name, 'coverage of test rows:', round(float(np.mean(~np.isnan(p[is_test]))), 4), flush=True)


def bucket(boot, da, mask):
    v = pd.Series(da).reindex(pd.DatetimeIndex(boot.days)).to_numpy(dtype='float64')
    ok = (~np.isnan(v)) & mask
    a = np.where(ok, v, 0).astype(np.float32)
    dr = (boot.W @ a) / np.maximum(boot.W @ ok.astype(np.float32), 1e-9)
    lo, hi = np.percentile(dr, [2.5, 97.5])
    return float(v[ok].mean()), float(lo), float(hi), int(ok.sum())


def evaluate():
    panel = C.load_panel()
    tf = C.test_frame(panel)
    boot = C.Boot(tf, n_boot=10000)
    y = tf['y'].to_numpy(dtype='float64')
    rows, deltas, bks = [], [], []
    gz = tf.groupby('Date')['gap_z60'].first().reindex(pd.DatetimeIndex(boot.days)).to_numpy(dtype='float64')
    buckets = {'|g|<0.5': np.abs(gz) < 0.5, '0.5<=|g|<1.5': (np.abs(gz) >= 0.5) & (np.abs(gz) < 1.5), '|g|>=1.5': np.abs(gz) >= 1.5}
    for model in ('lgb', 'lr'):
        for name in CONFIGS:
            ps, ths = C.load_pred(f'stock_{model}_{name}')
            pg, thg = C.load_pred(f'{model}_{name}')
            cov = ~np.isnan(ps)
            pg_c = np.where(cov, pg, np.nan)                     # paired comparison on the rows the stock models cover
            es = C.evaluate(tf[cov].reset_index(drop=True), ps[cov], ths[cov])
            eg = C.evaluate(tf[cov].reset_index(drop=True), pg[cov], thg[cov])
            rep = REPORTED[name] if model == 'lgb' else {}
            for mode, e in (('stock', es), ('global', eg)):
                rows.append({'model': model, 'config': name, 'mode': mode, 'coverage': float(cov.mean()),
                             'auroc_yavg': e['auroc_yavg'], 'auroc_pooled': e['auroc_pooled'], 'perday': e['perday_pooled'],
                             'perday_yavg': e['perday_yavg'], 'auroc_on_yavg': e['auroc_on_yavg'], 'auroc_id_yavg': e['auroc_id_yavg'],
                             'accuracy_yavg': e['accuracy_yavg'], 'macro_f1_yavg': e['macro_f1_yavg'],
                             'reported_auroc': rep.get('auroc', np.nan) if mode == 'stock' else np.nan,
                             'reported_perday': rep.get('perday', np.nan) if mode == 'stock' else np.nan,
                             'reported_on': rep.get('on', np.nan) if mode == 'stock' else np.nan,
                             'reported_id': rep.get('id', np.nan) if mode == 'stock' else np.nan})
            a_pt, a_dr = boot.pooled(y, ps)
            b_pt, b_dr = boot.pooled(y, pg_c)
            deltas.append({'model': model, 'config': name, 'metric': 'pooled AUROC', 'stock': a_pt, 'global': b_pt, **C.delta_stats(a_dr, b_dr, a_pt, b_pt)})
            das, dag = C.day_auc(tf, ps), C.day_auc(tf, pg_c)
            a_pt, a_dr = boot.perday(das)
            b_pt, b_dr = boot.perday(dag)
            deltas.append({'model': model, 'config': name, 'metric': 'per-day AUC', 'stock': a_pt, 'global': b_pt, **C.delta_stats(a_dr, b_dr, a_pt, b_pt)})
            for lab, mk in buckets.items():
                v, lo, hi, n = bucket(boot, das, mk)
                bks.append({'model': f'stock_{model}_{name}', 'bucket': lab, 'days': n, 'perday_auc': v, 'lo': lo, 'hi': hi})
    m1 = C.load_pred('spec2_M1_rho_x_gap')[0]
    da1 = C.day_auc(tf, m1)
    for lab, mk in buckets.items():
        v, lo, hi, n = bucket(boot, da1, mk)
        bks.append({'model': 'M1_rho_x_gap (2 parameters)', 'bucket': lab, 'days': n, 'perday_auc': v, 'lo': lo, 'hi': hi})
    # 100 per-stock VIXPRE models vs the two-parameter M1, per-day AUC on the rows both cover
    for model in ('lgb', 'lr'):
        ps = C.load_pred(f'stock_{model}_VIXPRE')[0]
        cov = ~np.isnan(ps) & ~np.isnan(m1)
        a_pt, a_dr = boot.perday(C.day_auc(tf, np.where(cov, ps, np.nan)))
        b_pt, b_dr = boot.perday(C.day_auc(tf, np.where(cov, m1, np.nan)))
        deltas.append({'model': model, 'config': 'stock VIXPRE -> M1', 'metric': 'per-day AUC', 'stock': a_pt, 'global': b_pt, **C.delta_stats(a_dr, b_dr, a_pt, b_pt)})
    rows, deltas, bks = pd.DataFrame(rows), pd.DataFrame(deltas), pd.DataFrame(bks)
    rows.to_csv(os.path.join(C.RES, 'supp5_stock_summary.csv'), index=False)
    deltas.to_csv(os.path.join(C.RES, 'supp5_stock_vs_global.csv'), index=False)
    bks.to_csv(os.path.join(C.RES, 'supp5_stock_buckets.csv'), index=False)
    pd.set_option('display.width', 250)
    print(rows.round(4).to_string(index=False))
    print(deltas.round(4).to_string(index=False))
    print(bks.round(4).to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'fit' and sys.argv[2] in ('lgb', 'lr'):
        fit(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == 'eval':
        evaluate()
    else:
        sys.exit(__doc__)
