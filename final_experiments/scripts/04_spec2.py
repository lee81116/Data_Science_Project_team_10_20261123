"""Spec 2 - timing x selection decomposition. Structural model: logit(p) = a + b * (rho_i * gap). Global, train 2017+."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def run_custom(panel, build, model='lr'):
    """build(panel, train_mask) -> feature matrix for all rows (lets a fold add train-only statistics)."""
    y_all = panel['y'].to_numpy(dtype='float64')
    is_test = (panel['Date'] >= '2018-01-01').to_numpy()
    p = np.full(len(panel), np.nan); thr = np.full(len(panel), np.nan)
    for yr, tr, te in C.folds(panel, '2017-01-01'):
        x = build(panel, tr)
        ok = tr & ~np.isnan(x).all(axis=1)
        p[te] = C.fit_predict(x[ok], y_all[ok], x[te], model)
        thr[te] = y_all[ok].mean()
    return p[is_test], thr[is_test]


def bucket_stats(boot, da, mask_days, label):
    v = pd.Series(da).reindex(pd.DatetimeIndex(boot.days)).to_numpy(dtype='float64')
    ok = (~np.isnan(v)) & mask_days
    a = np.where(ok, v, 0).astype(np.float32)
    draws = (boot.W @ a) / np.maximum(boot.W @ ok.astype(np.float32), 1e-9)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {'bucket': label, 'days': int(ok.sum()), 'perday_auc': float(v[ok].mean()), 'lo': float(lo), 'hi': float(hi)}, draws


def main():
    panel = C.load_panel()
    panel['rho_gap'] = panel['rho'] * panel['gap_z60']
    tf = C.test_frame(panel)
    boot = C.Boot(tf, n_boot=10000)

    def up_rate(pn, tr):
        u = pn.loc[tr].groupby('Ticker')['y'].mean()
        col = pn['Ticker'].map(u).fillna(pn.loc[tr, 'y'].mean()).to_numpy(dtype='float64')
        return np.column_stack([pn['gap_next'].to_numpy(dtype='float64'), pn['gap_z60'].to_numpy(dtype='float64'), col])

    preds = {'M0_gap': C.load_pred('lr_VIXPRE')[0]}
    jobs = {
        'M1_rho_x_gap': (lambda pn, tr: pn[['rho_gap']].to_numpy(dtype='float64'), 'lr'),
        'M2_gap_rho_inter': (lambda pn, tr: pn[['gap_z60', 'rho', 'rho_gap']].to_numpy(dtype='float64'), 'lr'),
        'M3_gap_baserate': (up_rate, 'lr'),
        'M4_lgb_T27_VIXPRE_rho': (lambda pn, tr: pn[C.T27 + C.VIXPRE + ['rho']].to_numpy(dtype='float64'), 'lgb'),
        'M5_baserate_only': (lambda pn, tr: up_rate(pn, tr)[:, [2]], 'lr'),
    }
    for name, (build, model) in jobs.items():
        p, thr = run_custom(panel, build, model)
        C.save_pred(f'spec2_{name}', p, thr)
        preds[name] = p
        print(name, 'done', flush=True)
    preds['REF_lgb_T27+VIXPRE'] = C.load_pred('lgb_T27+VIXPRE_common')[0]
    preds['REF_lgb_T27'] = C.load_pred('lgb_T27_common')[0]
    preds['RHO_only'] = tf['rho'].to_numpy(dtype='float64')

    # rho descriptive stats in the test period
    g = tf.groupby('Date')['rho']
    desc = {'rho_mean': float(tf['rho'].mean()), 'rho_cs_sd_avg': float(g.std().mean()), 'rho_p10': float(tf['rho'].quantile(.1)),
            'rho_p90': float(tf['rho'].quantile(.9)), 'rho_missing_share': float(tf['rho'].isna().mean())}
    print(desc)

    rows, cache = [], {}
    for nm, p in preds.items():
        r = {'model': nm}
        for lab in ('y', 'y_on', 'y_id'):
            pt, dr = boot.pooled(tf[lab].to_numpy(dtype='float64'), p)
            lo, hi = np.percentile(dr, [2.5, 97.5])
            r[f'auroc_{lab}'] = pt; r[f'auroc_{lab}_lo'] = lo; r[f'auroc_{lab}_hi'] = hi
            cache[(nm, 'pooled', lab)] = (pt, dr)
        da = C.day_auc(tf, p, 'y')
        pt, dr = boot.perday(da)
        lo, hi = np.percentile(dr, [2.5, 97.5])
        r['perday_y'] = pt; r['perday_y_lo'] = lo; r['perday_y_hi'] = hi
        cache[(nm, 'perday', 'y')] = (pt, dr)
        cache[(nm, 'da')] = da
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.RES, 'spec2_metrics.csv'), index=False)

    pairs = [('M0_gap', 'M1_rho_x_gap', 'pooled'), ('M0_gap', 'M1_rho_x_gap', 'perday'), ('M1_rho_x_gap', 'M2_gap_rho_inter', 'pooled'),
             ('M1_rho_x_gap', 'M2_gap_rho_inter', 'perday'), ('M0_gap', 'M3_gap_baserate', 'pooled'), ('M0_gap', 'M3_gap_baserate', 'perday'),
             ('REF_lgb_T27+VIXPRE', 'M4_lgb_T27_VIXPRE_rho', 'pooled'), ('REF_lgb_T27+VIXPRE', 'M4_lgb_T27_VIXPRE_rho', 'perday'),
             ('M3_gap_baserate', 'M2_gap_rho_inter', 'perday')]
    dl = []
    for a, b, met in pairs:
        A, B = cache[(a, met, 'y')], cache[(b, met, 'y')]
        dl.append({'a': a, 'b': b, 'metric': met, **C.delta_stats(A[1], B[1], A[0], B[0])})
    dl = pd.DataFrame(dl)
    dl.to_csv(os.path.join(C.RES, 'spec2_deltas.csv'), index=False)

    # gap-size buckets and sign-flip test
    gz = tf.groupby('Date')['gap_z60'].first().reindex(pd.DatetimeIndex(boot.days)).to_numpy(dtype='float64')
    buckets = {'|g|<0.5': np.abs(gz) < 0.5, '0.5<=|g|<1.5': (np.abs(gz) >= 0.5) & (np.abs(gz) < 1.5), '|g|>=1.5': np.abs(gz) >= 1.5,
               'g>0 (VIX up)': gz > 0, 'g<0 (VIX down)': gz < 0, 'g>=+1.5': gz >= 1.5, 'g<=-1.5': gz <= -1.5}
    bk = []
    for nm in ('M1_rho_x_gap', 'M2_gap_rho_inter', 'M3_gap_baserate', 'M5_baserate_only', 'M4_lgb_T27_VIXPRE_rho', 'REF_lgb_T27+VIXPRE', 'REF_lgb_T27', 'RHO_only'):
        for lab, mk in buckets.items():
            st, _ = bucket_stats(boot, cache[(nm, 'da')], mk, lab)
            bk.append({'model': nm, **st})
    bk = pd.DataFrame(bk)
    bk.to_csv(os.path.join(C.RES, 'spec2_buckets.csv'), index=False)
    pd.Series(desc).to_csv(os.path.join(C.RES, 'spec2_rho_stats.csv'))

    pd.set_option('display.width', 250)
    print(res[['model', 'auroc_y', 'auroc_y_lo', 'auroc_y_hi', 'perday_y', 'perday_y_lo', 'perday_y_hi', 'auroc_y_on', 'auroc_y_id']].round(4).to_string(index=False))
    print(dl.round(4).to_string(index=False))
    print(bk.pivot(index='model', columns='bucket', values='perday_auc')[list(buckets)].round(4).to_string())
    print(bk[bk.model == 'RHO_only'].round(4).to_string(index=False))
    print(bk[bk.model == 'M1_rho_x_gap'].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
