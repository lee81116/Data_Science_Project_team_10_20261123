"""Supp S6 - decomposed model at the 03:15 ET prediction time.
  timing x exposure : rho_i * gap_z60                     (same term as M1 in Spec 2)
  selection         : LightGBM on T27 trained on the cross-sectional label (beat the same-day median), used as an
                      out-of-sample score, logit-transformed and demeaned within each day so it carries no market-level information
  combiner          : logistic regression on [rho*gap, selection]  (3 parameters), walk-forward, trained on 2017+ rows
The selection score of year Y comes from a model trained on data before Y, so the combiner never sees in-sample scores.
Usage: python 09_supp_decomp.py fit | eval
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SEL_PATH = os.path.join(C.PRED, 'supp6_selection_oos_2017plus.npy')


def selection_scores(panel):
    """Walk-forward OOS selection probability for every row dated 2017 or later (NaN before)."""
    med = panel.groupby('Date')['fut_ret'].transform('median')
    y_cs = (panel['fut_ret'] > med).astype('float64').to_numpy()
    x = panel[C.T27].to_numpy(dtype='float64')
    d = panel['Date']
    p = np.full(len(panel), np.nan)
    for yr in range(2017, 2027):
        te = (d.dt.year == yr).to_numpy()
        tr = d < pd.Timestamp(yr, 1, 1)
        tr &= d < d[tr].max()                                  # 1-day purge
        tr = tr.to_numpy()
        p[te] = C.fit_predict(x[tr], y_cs[tr], x[te], 'lgb')
        print(f'  selection fold {yr}: train {tr.sum():,} test {te.sum():,}', flush=True)
    return p, y_cs


def fit():
    panel = C.load_panel()
    p_sel, y_cs = selection_scores(panel)
    np.save(SEL_PATH, p_sel)
    z = np.log(np.clip(p_sel, 1e-6, 1 - 1e-6) / (1 - np.clip(p_sel, 1e-6, 1 - 1e-6)))
    panel['sel'] = z - pd.Series(z).groupby(panel['Date'].to_numpy()).transform('mean').to_numpy()
    panel['rho_gap'] = panel['rho'] * panel['gap_z60']
    y_all = panel['y'].to_numpy(dtype='float64')
    is_test = (panel['Date'] >= '2018-01-01').to_numpy()
    jobs = {'D0_rho_gap': ['rho_gap'], 'D1_rho_gap+sel': ['rho_gap', 'sel'], 'D2_gap+rho_gap+sel': ['gap_z60', 'rho_gap', 'sel'], 'SEL_only': ['sel']}
    coefs = []
    for name, feats in jobs.items():
        x = panel[feats].to_numpy(dtype='float64')
        p = np.full(len(panel), np.nan); thr = np.full(len(panel), np.nan)
        for yr, tr, te in C.folds(panel, '2017-01-01'):
            ok = tr & ~np.isnan(x).any(axis=1)
            p[te] = C.fit_predict(x[ok], y_all[ok], x[te], 'lr')
            thr[te] = y_all[ok].mean()
            if name == 'D1_rho_gap+sel':                      # near-unregularised fit on raw units, for reading the coefficients
                m = LogisticRegression(C=1e6, max_iter=2000).fit(x[ok], y_all[ok])
                coefs.append({'test_year': yr, 'train_rows': int(ok.sum()), 'intercept': float(m.intercept_[0]),
                              'b_rho_gap': float(m.coef_[0][0]), 'c_selection': float(m.coef_[0][1])})
        C.save_pred(f'supp6_{name}', p[is_test], thr[is_test])
        print(name, 'done', flush=True)
    pd.DataFrame(coefs).to_csv(os.path.join(C.RES, 'supp6_combiner_coefs.csv'), index=False)
    print(pd.DataFrame(coefs).round(4).to_string(index=False))


def evaluate():
    panel = C.load_panel()
    tf = C.test_frame(panel)
    boot = C.Boot(tf, n_boot=10000)
    preds = {k: C.load_pred(f'supp6_{k}') for k in ('D0_rho_gap', 'D1_rho_gap+sel', 'D2_gap+rho_gap+sel', 'SEL_only')}
    preds['M1 (Spec 2, saved)'] = C.load_pred('spec2_M1_rho_x_gap')
    preds['M4 LGB T27+VIXPRE+rho'] = C.load_pred('spec2_M4_lgb_T27_VIXPRE_rho')
    preds['REF LGB T27+VIXPRE'] = C.load_pred('lgb_T27+VIXPRE_common')
    preds['REF LGB T27 (16:00)'] = C.load_pred('lgb_T27_common')
    rows, cache = [], {}
    for nm, (p, thr) in preds.items():
        r = {'model': nm}
        for lab in ('y', 'y_on', 'y_id'):
            pt, dr = boot.pooled(tf[lab].to_numpy(dtype='float64'), p)
            lo, hi = np.percentile(dr, [2.5, 97.5])
            r[f'auroc_{lab}'] = pt; r[f'auroc_{lab}_lo'] = lo; r[f'auroc_{lab}_hi'] = hi
            cache[(nm, 'pooled', lab)] = (pt, dr)
        da = C.day_auc(tf, p)
        pt, dr = boot.perday(da)
        lo, hi = np.percentile(dr, [2.5, 97.5])
        r['perday'] = pt; r['perday_lo'] = lo; r['perday_hi'] = hi
        cache[(nm, 'perday', 'y')] = (pt, dr); cache[(nm, 'da')] = da
        e = C.evaluate(tf, p, thr)
        r['accuracy_yavg'] = e['accuracy_yavg']; r['macro_f1_yavg'] = e['macro_f1_yavg']
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.RES, 'supp6_metrics.csv'), index=False)
    pairs = [('D0_rho_gap', 'D1_rho_gap+sel'), ('D1_rho_gap+sel', 'D2_gap+rho_gap+sel'), ('REF LGB T27+VIXPRE', 'D1_rho_gap+sel'),
             ('M4 LGB T27+VIXPRE+rho', 'D1_rho_gap+sel'), ('REF LGB T27+VIXPRE', 'D0_rho_gap')]
    dl = []
    for a, b in pairs:
        for met in ('pooled', 'perday'):
            A, B = cache[(a, met, 'y')], cache[(b, met, 'y')]
            dl.append({'a': a, 'b': b, 'metric': met, **C.delta_stats(A[1], B[1], A[0], B[0])})
    dl = pd.DataFrame(dl)
    dl['p_holm'] = C.holm(dl['p'].to_numpy())
    dl.to_csv(os.path.join(C.RES, 'supp6_deltas.csv'), index=False)
    gz = tf.groupby('Date')['gap_z60'].first().reindex(pd.DatetimeIndex(boot.days)).to_numpy(dtype='float64')
    buckets = {'|g|<0.5': np.abs(gz) < 0.5, '0.5<=|g|<1.5': (np.abs(gz) >= 0.5) & (np.abs(gz) < 1.5), '|g|>=1.5': np.abs(gz) >= 1.5}
    bk = []
    for nm in ('D0_rho_gap', 'D1_rho_gap+sel', 'SEL_only', 'REF LGB T27+VIXPRE'):
        v = pd.Series(cache[(nm, 'da')]).reindex(pd.DatetimeIndex(boot.days)).to_numpy(dtype='float64')
        for lab, mk in buckets.items():
            ok = (~np.isnan(v)) & mk
            a = np.where(ok, v, 0).astype(np.float32)
            dr = (boot.W @ a) / np.maximum(boot.W @ ok.astype(np.float32), 1e-9)
            lo, hi = np.percentile(dr, [2.5, 97.5])
            bk.append({'model': nm, 'bucket': lab, 'days': int(ok.sum()), 'perday_auc': float(v[ok].mean()), 'lo': float(lo), 'hi': float(hi)})
    bk = pd.DataFrame(bk)
    bk.to_csv(os.path.join(C.RES, 'supp6_buckets.csv'), index=False)
    pd.set_option('display.width', 250)
    print(res.round(4).to_string(index=False))
    print(dl.round(4).to_string(index=False))
    print(bk.round(4).to_string(index=False))
    d0, m1 = preds['D0_rho_gap'][0], preds['M1 (Spec 2, saved)'][0]
    print('D0 vs saved M1: max abs diff', float(np.nanmax(np.abs(d0 - m1))))


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'fit':
        fit()
    elif len(sys.argv) >= 2 and sys.argv[1] == 'eval':
        evaluate()
    else:
        sys.exit(__doc__)
