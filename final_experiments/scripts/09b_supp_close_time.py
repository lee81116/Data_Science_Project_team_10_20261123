"""Supp S6b - the same decomposition at the 16:00 ET prediction time (no pre-market information).
  timing x exposure : rho_i * equal-weighted market return of day t   (market-level reversal, known at the close)
  selection         : OOS cross-sectional score from 09_supp_decomp.py (trained on the beat-the-median label)
Also tests label design alone: cross-sectional label vs up/down label with identical features and model.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    panel = C.load_panel()
    sel_path = os.path.join(C.PRED, 'supp6_selection_oos_2017plus.npy')
    if not os.path.exists(sel_path):
        sys.exit('run 09_supp_decomp.py fit first')
    p_sel = np.clip(np.load(sel_path), 1e-6, 1 - 1e-6)
    z = np.log(p_sel / (1 - p_sel))
    panel['sel'] = z - pd.Series(z).groupby(panel['Date'].to_numpy()).transform('mean').to_numpy()
    panel['rho_mkt'] = panel['rho'] * panel['mkt_ret']
    y_all = panel['y'].to_numpy(dtype='float64')
    is_test = (panel['Date'] >= '2018-01-01').to_numpy()
    tf = C.test_frame(panel)
    boot = C.Boot(tf, n_boot=10000)
    jobs = {'E0_rho_mkt': ['rho_mkt'], 'E1_rho_mkt+sel': ['rho_mkt', 'sel'], 'E2_mkt+rho_mkt+sel': ['mkt_ret', 'rho_mkt', 'sel']}
    preds = {}
    for name, feats in jobs.items():
        x = panel[feats].to_numpy(dtype='float64')
        p = np.full(len(panel), np.nan); thr = np.full(len(panel), np.nan)
        for yr, tr, te in C.folds(panel, '2017-01-01'):
            ok = tr & ~np.isnan(x).any(axis=1)
            p[te] = C.fit_predict(x[ok], y_all[ok], x[te], 'lr')
            thr[te] = y_all[ok].mean()
        C.save_pred(f'supp6b_{name}', p[is_test], thr[is_test])
        preds[name] = (p[is_test], thr[is_test])
    preds['SEL_only (cs label, LGB T27)'] = C.load_pred('supp6_SEL_only')
    preds['RAW selection prob (cs label)'] = (np.load(sel_path)[is_test], None)
    preds['REF LGB T27 full history'] = C.load_pred('lgb_T27')
    preds['REF LGB T27 2017+'] = C.load_pred('lgb_T27_common')
    preds['zero-parameter: -market return(t)'] = (-tf['mkt_ret'].to_numpy(dtype='float64'), None)
    rows, cache = [], {}
    for nm, (p, thr) in preds.items():
        pt, dr = boot.pooled(tf['y'].to_numpy(dtype='float64'), p)
        lo, hi = np.percentile(dr, [2.5, 97.5])
        da = C.day_auc(tf, p)
        pt2, dr2 = boot.perday(da)
        lo2, hi2 = np.percentile(dr2, [2.5, 97.5])
        cache[(nm, 'pooled')] = (pt, dr); cache[(nm, 'perday')] = (pt2, dr2)
        r = {'model': nm, 'auroc': pt, 'auroc_lo': lo, 'auroc_hi': hi, 'perday': pt2, 'perday_lo': lo2, 'perday_hi': hi2}
        if thr is not None:
            e = C.evaluate(tf, p, thr)
            r['auroc_yavg'] = e['auroc_yavg']; r['accuracy_yavg'] = e['accuracy_yavg']
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.RES, 'supp6b_metrics.csv'), index=False)
    pairs = [('REF LGB T27 full history', 'E1_rho_mkt+sel'), ('REF LGB T27 2017+', 'E1_rho_mkt+sel'), ('E0_rho_mkt', 'E1_rho_mkt+sel'),
             ('REF LGB T27 full history', 'RAW selection prob (cs label)'), ('E1_rho_mkt+sel', 'E2_mkt+rho_mkt+sel')]
    dl = []
    for a, b in pairs:
        for met in ('pooled', 'perday'):
            A, B = cache[(a, met)], cache[(b, met)]
            dl.append({'a': a, 'b': b, 'metric': met, **C.delta_stats(A[1], B[1], A[0], B[0])})
    dl = pd.DataFrame(dl)
    dl['p_holm'] = C.holm(dl['p'].to_numpy())
    dl.to_csv(os.path.join(C.RES, 'supp6b_deltas.csv'), index=False)
    pd.set_option('display.width', 250)
    print(res.round(4).to_string(index=False))
    print(dl.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
