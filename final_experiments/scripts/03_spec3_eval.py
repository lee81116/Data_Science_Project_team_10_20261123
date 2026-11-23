"""Spec 3 - evaluation: comparability table inputs (top/flop-k accuracy, per-day AUC, pooled AUROC, long-short return)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def topflop(tf, p, k, boot):
    """Per-day top-k / flop-k picks. Returns accuracy vs y_cs, accuracy vs y, long-short return (with bootstrap CIs)."""
    d = pd.DataFrame({'Date': tf['Date'].to_numpy(), 'p': p, 'ycs': tf['y_cs'].to_numpy(), 'y': tf['y'].to_numpy(),
                      'r': tf['fut_ret'].to_numpy()}).dropna()
    d['rk_hi'] = d.groupby('Date')['p'].rank(method='first', ascending=False)
    d['rk_lo'] = d.groupby('Date')['p'].rank(method='first', ascending=True)
    top, flop = d[d['rk_hi'] <= k], d[d['rk_lo'] <= k]
    days = pd.DatetimeIndex(boot.days)
    def per_day(s):
        return s.reindex(days).fillna(0).to_numpy(dtype=np.float32)
    hit_cs = per_day(top.groupby('Date')['ycs'].sum() + (1 - flop['ycs']).groupby(flop['Date']).sum())
    hit_abs = per_day(top.groupby('Date')['y'].sum() + (1 - flop['y']).groupby(flop['Date']).sum())
    cnt = per_day(top.groupby('Date').size() + flop.groupby('Date').size())
    ls = (top.groupby('Date')['r'].mean() - flop.groupby('Date')['r'].mean()).reindex(days)
    ok = ls.notna().to_numpy().astype(np.float32)
    lsv = ls.fillna(0).to_numpy(dtype=np.float32)
    out = {}
    for name, num, den in (('acc_cs', hit_cs, cnt), ('acc_abs', hit_abs, cnt), ('ls_ret', lsv, ok)):
        draws = (boot.W @ num) / (boot.W @ den)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out[name] = float(num.sum() / den.sum()); out[name + '_lo'] = float(lo); out[name + '_hi'] = float(hi)
    return out


def median_split_acc(tf, p):
    d = pd.DataFrame({'Date': tf['Date'].to_numpy(), 'p': p, 'ycs': tf['y_cs'].to_numpy()}).dropna()
    pred = (d['p'] > d.groupby('Date')['p'].transform('median')).astype(float)
    return float((pred == d['ycs']).mean())


def main():
    tf = pd.read_pickle(os.path.join(C.DATA, 'spec3_test_frame.pkl'))
    yrs = tf['Date'].dt.year.to_numpy()
    boot = C.Boot(tf, n_boot=10000)
    preds = {}
    for lab, nm in (('y_cs', 'rf_S1_krauss_cs'), ('y', 'rf_S2_krauss_abs')):
        p = np.full(len(tf), np.nan)
        for yr in C.TEST_YEARS:
            p[yrs == yr] = np.load(os.path.join(C.PRED, f'rf_fold_{lab}_{yr}.npy'))
        preds[nm] = p
        C.save_pred(f'spec3_{nm}', p)
    for m in ('lgb', 'lr'):
        for nm in ('S1_krauss_cs', 'S1b_T27_cs', 'S2_krauss_abs', 'T27_abs', 'S3tree_krauss_mkt_abs'):
            preds[f'{m}_{nm}'] = C.load_pred(f'spec3_{m}_{nm}')[0]
    preds['lgb_S3_linear_mkt_abs'] = C.load_pred('spec3_lgb_S3_linear_mkt_abs')[0]
    # date-level baseline: previous-day market return alone (reversal sign), for reference
    preds['mkt_reversal_only'] = -tf['mkt_ret'].to_numpy(dtype='float64')

    ycs, y = tf['y_cs'].to_numpy(dtype='float64'), tf['y'].to_numpy(dtype='float64')
    rows, cache = [], {}
    for nm, p in preds.items():
        r = {'model': nm}
        for lab, yy in (('cs', ycs), ('abs', y)):
            pt, dr = boot.pooled(yy, p)
            lo, hi = np.percentile(dr, [2.5, 97.5])
            r[f'auroc_{lab}'] = pt; r[f'auroc_{lab}_lo'] = lo; r[f'auroc_{lab}_hi'] = hi
            cache[(nm, 'pooled', lab)] = (pt, dr)
            pt, dr = boot.perday(C.day_auc(tf, p, 'y_cs' if lab == 'cs' else 'y'))
            lo, hi = np.percentile(dr, [2.5, 97.5])
            r[f'perday_{lab}'] = pt; r[f'perday_{lab}_lo'] = lo; r[f'perday_{lab}_hi'] = hi
            cache[(nm, 'perday', lab)] = (pt, dr)
        r['acc_full_median_split'] = median_split_acc(tf, p)
        for k in (2, 10):
            for kk, vv in topflop(tf, p, k, boot).items():
                r[f'k{k}_{kk}'] = vv
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.RES, 'spec3_metrics.csv'), index=False)

    pairs = [('lgb_T27_abs', 'lgb_S2_krauss_abs', 'pooled', 'abs'), ('lgb_S2_krauss_abs', 'lgb_S3_linear_mkt_abs', 'pooled', 'abs'),
             ('lgb_S2_krauss_abs', 'lgb_S3_linear_mkt_abs', 'perday', 'abs'), ('lgb_S2_krauss_abs', 'lgb_S3tree_krauss_mkt_abs', 'pooled', 'abs'),
             ('lgb_S1_krauss_cs', 'lgb_S2_krauss_abs', 'perday', 'cs'), ('lgb_S1_krauss_cs', 'rf_S1_krauss_cs', 'perday', 'cs'),
             ('lgb_S1_krauss_cs', 'lgb_S1b_T27_cs', 'perday', 'cs'), ('lr_T27_abs', 'lr_S2_krauss_abs', 'pooled', 'abs')]
    dl = []
    for a, b, met, lab in pairs:
        A, B = cache[(a, met, lab)], cache[(b, met, lab)]
        dl.append({'a': a, 'b': b, 'metric': met, 'label': lab, **C.delta_stats(A[1], B[1], A[0], B[0])})
    dl = pd.DataFrame(dl)
    dl.to_csv(os.path.join(C.RES, 'spec3_deltas.csv'), index=False)

    by = []
    for nm in ('rf_S1_krauss_cs', 'lgb_S1_krauss_cs'):
        da = C.day_auc(tf, preds[nm], 'y_cs')
        by.append(da.groupby(da.index.year).mean().rename(nm))
    pd.concat(by, axis=1).to_csv(os.path.join(C.RES, 'spec3_perday_cs_by_year.csv'))

    pd.set_option('display.width', 250)
    show = ['model', 'auroc_abs', 'perday_abs', 'auroc_cs', 'perday_cs', 'acc_full_median_split', 'k2_acc_cs', 'k10_acc_cs', 'k10_acc_cs_lo', 'k10_acc_cs_hi', 'k10_ls_ret', 'k2_ls_ret']
    t = res[show].copy()
    for c in ('k10_ls_ret', 'k2_ls_ret'):
        t[c] = t[c] * 100
    print(t.round(4).to_string(index=False))
    print(dl.round(4).to_string(index=False))
    print(pd.concat(by, axis=1).round(4).to_string())


if __name__ == '__main__':
    main()
