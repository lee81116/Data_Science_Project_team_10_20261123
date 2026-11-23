"""RQ1 model families: Random Forest and XGBoost beside logistic regression and LightGBM.

Global mode, yearly walk-forward, fixed hyper-parameters (no tuning), same protocol as common.run_config.
The teammate's v2 design lists LR, RF, XGBoost and LightGBM for RQ1; only LR and LightGBM had been run.

Usage:
  python3 18_rq1_models.py fit rf  T27 [years...]
  python3 18_rq1_models.py fit xgb T27+VIXPRE
  python3 18_rq1_models.py eval

Per-fold predictions are cached in preds/rq1_folds/, so an interrupted fit resumes where it stopped.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CONFIGS = {'T27': C.T27, 'T27+VIXPRE': C.T27 + C.VIXPRE}
# Capacity kept close to the LightGBM setting (7 leaves, 200 trees, min_child 1000, lr 0.03, l2 10).
RF_PARAMS = dict(n_estimators=300, max_depth=8, min_samples_leaf=1000, max_features='sqrt',
                 max_samples=0.3, n_jobs=4, random_state=42)
XGB_PARAMS = dict(n_estimators=200, learning_rate=0.03, max_depth=3, min_child_weight=250,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, tree_method='hist',
                  n_jobs=4, random_state=42, eval_metric='logloss')
FOLD_DIR = os.path.join(C.PRED, 'rq1_folds')
# Written down before the first run (2026-11-23).
EXPECT = {
    'T27: |model - LightGBM| pooled AUROC': 'below 0.005 for RF and XGBoost',
    'T27 level': 'RF and XGBoost pooled AUROC between 0.505 and 0.520',
    'T27 -> T27+VIXPRE': 'pooled gain between +0.04 and +0.07 for RF and XGBoost, interval excludes 0',
}


def fit_predict_extra(xtr, ytr, xte, model):
    med, lo, hi = C._stats(xtr)
    xtr, xte = np.clip(xtr, lo, hi), np.clip(xte, lo, hi)
    if model == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        if np.isnan(med).any():
            raise ValueError('a feature is entirely missing in the training rows')
        xtr = np.where(np.isnan(xtr), med, xtr)
        xte = np.where(np.isnan(xte), med, xte)
        m = RandomForestClassifier(**RF_PARAMS).fit(xtr, ytr)
        return m.predict_proba(xte)[:, 1]
    if model == 'xgb':
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise SystemExit('xgboost is not installed: pip install xgboost') from e
        m = XGBClassifier(**XGB_PARAMS).fit(xtr, ytr)          # NaN handled natively
        return m.predict_proba(xte)[:, 1]
    raise ValueError(model)


def fold_path(model, config, yr):
    return os.path.join(FOLD_DIR, f'{model}_{config}_{yr}.npz')


def fit(model, config, years):
    os.makedirs(FOLD_DIR, exist_ok=True)
    panel = C.load_panel()
    feats = CONFIGS[config]
    x_all = panel[feats].to_numpy(dtype='float64')
    y_all = panel['y'].to_numpy(dtype='float64')
    ok = ~np.isnan(y_all)
    for yr, tr, te in C.folds(panel):
        if years and yr not in years:
            continue
        path = fold_path(model, config, yr)
        if os.path.exists(path):
            print(f'  {model} {config} {yr}: cached', flush=True)
            continue
        tr = tr & ok
        t0 = time.time()
        p = fit_predict_extra(x_all[tr], y_all[tr], x_all[te], model)
        np.savez(path, p=p, thr=y_all[tr].mean(), n_train=int(tr.sum()))
        print(f'  {model} {config} {yr}: train {tr.sum():,} test {te.sum():,} {time.time() - t0:.0f}s', flush=True)


def assemble(panel, model, config):
    p = np.full(len(panel), np.nan)
    thr = np.full(len(panel), np.nan)
    for yr, _, te in C.folds(panel):
        path = fold_path(model, config, yr)
        if not os.path.exists(path):
            raise FileNotFoundError(f'missing fold: {path}')
        z = np.load(path)
        p[te] = z['p']
        thr[te] = float(z['thr'])
    is_test = (panel['Date'] >= '2018-01-01').to_numpy()
    return p[is_test], thr[is_test]


def ci(draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate():
    panel = C.load_panel()
    tf = C.test_frame(panel)
    y = tf['y'].to_numpy(dtype='float64')
    boot = C.Boot(tf, n_boot=5000)
    preds = {}
    for config in CONFIGS:
        for model in ('lr', 'lgb'):
            preds[(model, config)] = C.load_pred(f'{model}_{config}')
        for model in ('rf', 'xgb'):
            p, thr = assemble(panel, model, config)
            C.save_pred(f'{model}_{config}', p, thr)
            preds[(model, config)] = (p, thr)

    rows, by_year, draws = [], [], {}
    for (model, config), (p, thr) in preds.items():
        ev = C.evaluate(tf, p, thr)
        b = ev.pop('by_year')
        b.insert(0, 'config', config)
        b.insert(0, 'model', model)
        by_year.append(b)
        pt, dr = boot.pooled(y, p)
        da = C.day_auc(tf, p)
        ppt, pdr = boot.perday(da)
        draws[(model, config)] = (pt, dr, ppt, pdr)
        lo, hi = ci(dr)
        plo, phi = ci(pdr)
        rows.append({'model': model, 'config': config, 'auroc_pooled': pt, 'auroc_lo': lo, 'auroc_hi': hi,
                     'auroc_yavg': ev['auroc_yavg'], 'perday': ppt, 'perday_lo': plo, 'perday_hi': phi,
                     'accuracy_yavg': ev.get('accuracy_yavg'), 'macro_f1_yavg': ev.get('macro_f1_yavg'),
                     'pr_auc_yavg': ev.get('pr_auc_yavg'), 'auroc_on_yavg': ev.get('auroc_on_yavg'),
                     'auroc_id_yavg': ev.get('auroc_id_yavg'),
                     'auroc_year_min': float(b['auroc'].min()), 'auroc_year_max': float(b['auroc'].max())})
    met = pd.DataFrame(rows)

    drows = []
    for config in CONFIGS:
        a = draws[('lgb', config)]
        for model in ('lr', 'rf', 'xgb'):
            b = draws[(model, config)]
            d = C.delta_stats(a[1], b[1], a[0], b[0])
            dp = C.delta_stats(a[3], b[3], a[2], b[2])
            drows.append({'comparison': f'{model} - lgb', 'config': config, 'd_pooled': d['delta'], 'lo': d['lo'],
                          'hi': d['hi'], 'p': d['p'], 'd_perday': dp['delta'], 'perday_lo': dp['lo'],
                          'perday_hi': dp['hi'], 'perday_p': dp['p']})
    for model in ('lr', 'lgb', 'rf', 'xgb'):
        a, b = draws[(model, 'T27')], draws[(model, 'T27+VIXPRE')]
        d = C.delta_stats(a[1], b[1], a[0], b[0])
        dp = C.delta_stats(a[3], b[3], a[2], b[2])
        drows.append({'comparison': f'{model}: T27 -> T27+VIXPRE', 'config': 'both', 'd_pooled': d['delta'],
                      'lo': d['lo'], 'hi': d['hi'], 'p': d['p'], 'd_perday': dp['delta'], 'perday_lo': dp['lo'],
                      'perday_hi': dp['hi'], 'perday_p': dp['p']})
    dl = pd.DataFrame(drows)
    dl['p_holm'] = C.holm(dl['p'].to_numpy())

    met.to_csv(os.path.join(C.RES, 'rq1_models_metrics.csv'), index=False)
    dl.to_csv(os.path.join(C.RES, 'rq1_models_deltas.csv'), index=False)
    pd.concat(by_year).to_csv(os.path.join(C.RES, 'rq1_models_by_year.csv'), index=False)
    with open(os.path.join(C.RES, 'rq1_models_setup.json'), 'w') as f:
        json.dump({'rf_params': RF_PARAMS, 'xgb_params': XGB_PARAMS, 'lgb_params': C.LGB_PARAMS,
                   'expectations_written_before_run': EXPECT, 'n_boot': 5000, 'block_days': 20,
                   'test_rows': int(len(tf)), 'test_days': int(boot.D)}, f, indent=2, ensure_ascii=False)
    pd.set_option('display.width', 250)
    print(met.round(4).to_string(index=False))
    print()
    print(dl.round(4).to_string(index=False))
    yr = pd.concat(by_year)
    print()
    print(yr[yr['config'] == 'T27'].pivot(index='year', columns='model', values='auroc').round(4).to_string())


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('fit', 'eval'):
        raise SystemExit(__doc__)
    if sys.argv[1] == 'eval':
        evaluate()
        return
    if len(sys.argv) < 4 or sys.argv[2] not in ('rf', 'xgb') or sys.argv[3] not in CONFIGS:
        raise SystemExit(__doc__)
    fit(sys.argv[2], sys.argv[3], [int(a) for a in sys.argv[4:]])


if __name__ == '__main__':
    main()
