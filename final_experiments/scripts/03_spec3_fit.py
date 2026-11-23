"""Spec 3 - literature-comparable benchmark (Krauss, Do & Huck 2017 features and label), three steps.

Usage: 03_spec3_fit.py lgb|lr            -> all folds for the fast models
       03_spec3_fit.py rf 2018 2019 ...  -> random forest, selected test years (cached per year)
All rows need the full 240-day history, so every model here (and T27 for comparison) uses the same subset.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

RF_PARAMS = dict(n_estimators=200, max_depth=10, min_samples_leaf=200, max_features='sqrt',
                 max_samples=0.2, n_jobs=4, random_state=42)


def spec3_panel():
    panel = C.load_panel()
    panel = panel[panel[C.KRAUSS].notna().all(axis=1)].reset_index(drop=True)
    g = panel.groupby('Date')['fut_ret']
    med, cnt = g.transform('median'), g.transform('size')
    panel['y_cs'] = (panel['fut_ret'] > med).astype('float32').where(cnt >= 30)
    return panel


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def run_rf(panel, label, years):
    x_all = panel[C.KRAUSS].to_numpy(dtype='float32')
    y_all = panel[label].to_numpy(dtype='float64')
    ok = ~np.isnan(y_all)
    for yr, tr, te in C.folds(panel):
        if yr not in years:
            continue
        path = os.path.join(C.PRED, f'rf_fold_{label}_{yr}.npy')
        if os.path.exists(path):
            continue
        m = RandomForestClassifier(**RF_PARAMS).fit(x_all[tr & ok], y_all[tr & ok])
        np.save(path, m.predict_proba(x_all[te])[:, 1])
        print(f'  rf {label} {yr} done', flush=True)


def step3(panel, verbose=False):
    """Step 2 LightGBM score + previous-day market return, combined by a 2-coefficient logistic regression."""
    import lightgbm as lgb
    x_all = panel[C.KRAUSS].to_numpy(dtype='float64')
    y_all = panel['y'].to_numpy(dtype='float64')
    mk = panel['mkt_ret'].to_numpy(dtype='float64')
    is_test = (panel['Date'] >= '2018-01-01').to_numpy()
    p = np.full(len(panel), np.nan)
    coefs = []
    for yr, tr, te in C.folds(panel):
        med, lo, hi = C._stats(x_all[tr])
        xtr, xte = np.clip(x_all[tr], lo, hi), np.clip(x_all[te], lo, hi)
        m = lgb.LGBMClassifier(**C.LGB_PARAMS).fit(xtr, y_all[tr])
        sd = np.nanstd(mk[tr])
        ztr = np.column_stack([logit(m.predict_proba(xtr)[:, 1]), mk[tr] / sd])
        zte = np.column_stack([logit(m.predict_proba(xte)[:, 1]), mk[te] / sd])
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(ztr, y_all[tr])
        p[te] = lr.predict_proba(zte)[:, 1]
        coefs.append({'year': yr, 'coef_logit': lr.coef_[0][0], 'coef_mkt_ret_z': lr.coef_[0][1]})
    return p[is_test], pd.DataFrame(coefs)


def main():
    which = sys.argv[1]
    panel = spec3_panel()
    tf_idx = panel['Date'] >= '2018-01-01'
    panel.loc[tf_idx, ['Ticker', 'Date', 'y', 'y_cs', 'y_on', 'y_id', 'fut_ret', 'mkt_ret', 'gap_z60', 'gap_next', 'on_ret', 'id_ret', 'rho']] \
        .reset_index(drop=True).to_pickle(os.path.join(C.DATA, 'spec3_test_frame.pkl'))
    print('spec3 panel rows', len(panel), '| test rows', int(tf_idx.sum()), '| y_cs mean', round(float(panel['y_cs'].mean()), 4), flush=True)
    if which == 'rf':
        years = [int(a) for a in sys.argv[2:]] or C.TEST_YEARS
        for label in ('y_cs', 'y'):
            run_rf(panel, label, years)
        return
    jobs = {  # name: (features, label)
        'S1_krauss_cs': (C.KRAUSS, 'y_cs'),
        'S1b_T27_cs': (C.T27, 'y_cs'),
        'S2_krauss_abs': (C.KRAUSS, 'y'),
        'T27_abs': (C.T27, 'y'),
        'S3tree_krauss_mkt_abs': (C.KRAUSS + ['mkt_ret'], 'y'),
    }
    for name, (feats, label) in jobs.items():
        p, thr = C.run_config(panel, feats, which, None, None, label=label, verbose=False)
        C.save_pred(f'spec3_{which}_{name}', p, thr)
        print(f'[{which}] {name} done', flush=True)
    if which == 'lgb':
        p, coefs = step3(panel)
        C.save_pred('spec3_lgb_S3_linear_mkt_abs', p)
        coefs.to_csv(os.path.join(C.RES, 'spec3_step3_coefs.csv'), index=False)
        print(coefs.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
