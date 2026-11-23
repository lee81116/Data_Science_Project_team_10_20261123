"""Prospective holdout, step 2 - models frozen on the Kaggle panel (data up to 2026-02-17), scored once on later dates.
  build : rebuild features for 2023+ from the Yahoo download with the same code as 00_build_panel.py, then check them against the existing panel on the overlap
  eval  : train every model once on the existing panel, predict the holdout rows (Date >= 2026-02-18), evaluate with a 20-day block bootstrap
Expectations were written to results/holdout_expectations.json before the download.
Usage: python 15_holdout_eval.py build | eval
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

HOLD = os.path.join(C.DATA, 'holdout')
BUILD = os.path.join(HOLD, 'build')
HP = os.path.join(BUILD, 'data', 'panel.pkl')
CUT = pd.Timestamp('2026-02-18')            # first date without a label in the Kaggle panel


def load_module(fname, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ci(d):
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def build():
    bp = load_module('00_build_panel.py', 'build_panel')
    bp.RAW = os.path.join(HOLD, 'yahoo_ohlcv.csv')
    bp.VIX = os.path.join(HOLD, 'vix_cboe_latest.csv')
    bp.ROOT = BUILD
    bp.main()
    hp = pd.read_pickle(HP)
    panel = C.load_panel()
    cols = C.T27 + ['rho', 'gap_z60', 'gap_next', 'fut_ret', 'on_ret', 'id_ret']
    m = panel.loc[panel['Date'] >= '2025-03-01', ['Ticker', 'Date'] + cols + ['y']].merge(hp[['Ticker', 'Date'] + cols + ['y']], on=['Ticker', 'Date'], suffixes=('_k', '_y'))
    rows = []
    for grp, d in (('all tickers except AZN', m[m['Ticker'] != 'AZN']), ('AZN only', m[m['Ticker'] == 'AZN'])):
        for c in cols:
            a, b = d[c + '_k'].to_numpy(dtype='float64'), d[c + '_y'].to_numpy(dtype='float64')
            ok = ~(np.isnan(a) | np.isnan(b))
            rows.append({'group': grp, 'feature': c, 'n': int(ok.sum()), 'corr': float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan,
                         'median_abs_diff': float(np.median(np.abs(a[ok] - b[ok]))), 'p99_abs_diff': float(np.percentile(np.abs(a[ok] - b[ok]), 99)),
                         'sd_kaggle': float(np.std(a[ok]))})
        rows.append({'group': grp, 'feature': 'label y agreement', 'n': int(len(d)), 'corr': float((d['y_k'] == d['y_y']).mean())})
    chk = pd.DataFrame(rows)
    chk.to_csv(os.path.join(C.RES, 'holdout_feature_check.csv'), index=False)
    hold = hp[hp['Date'] >= CUT]
    print('holdout rows', len(hold), '| days', hold['Date'].nunique(), '| tickers', hold['Ticker'].nunique(), '|', hold['Date'].min().date(), 'to', hold['Date'].max().date())
    print('overlap rows compared', len(m), '| existing panel rows since 2025-03-01', int((panel['Date'] >= '2025-03-01').sum()))
    pd.set_option('display.width', 220)
    x = chk[chk.group == 'all tickers except AZN']
    print(x.round(6).to_string(index=False))
    print(chk[(chk.group == 'AZN only') & chk.feature.isin(['return_1d', 'rho', 'label y agreement'])].round(4).to_string(index=False))


def evaluate():
    panel = C.load_panel()
    hp = pd.read_pickle(HP)
    hold = hp[hp['Date'] >= CUT].sort_values(['Date', 'Ticker']).reset_index(drop=True)
    if panel['Date'].max() >= CUT:
        sys.exit('the training panel reaches into the holdout period')
    d = panel['Date']
    tr_all = (d < d.max()).to_numpy()                                   # 1-day purge, as in the walk-forward protocol
    tr_17 = tr_all & (d >= '2017-01-01').to_numpy()
    y_tr = panel['y'].to_numpy(dtype='float64')
    med = panel.groupby('Date')['fut_ret'].transform('median')
    ycs_tr = (panel['fut_ret'] > med).astype('float64').to_numpy()
    hold['y_cs'] = (hold['fut_ret'] > hold.groupby('Date')['fut_ret'].transform('median')).astype('float64')
    panel['rho_gap'] = panel['rho'] * panel['gap_z60']; hold['rho_gap'] = hold['rho'] * hold['gap_z60']

    def fit(feats, mask, label=y_tr, model='lgb', xh=None, xt=None):
        xt = panel[feats].to_numpy(dtype='float64') if xt is None else xt
        xh = hold[feats].to_numpy(dtype='float64') if xh is None else xh
        ok = mask & ~np.isnan(label)
        if model == 'lr':
            ok &= ~np.isnan(xt).any(axis=1)
        return C.fit_predict(xt[ok], label[ok], xh, model), float(label[ok].mean())

    preds, thr = {}, {}
    preds['LGB T27 (16:00)'], thr['LGB T27 (16:00)'] = fit(C.T27, tr_all)
    preds['LGB T27+VIXPRE'], thr['LGB T27+VIXPRE'] = fit(C.T27 + C.VIXPRE, tr_all)
    preds['LGB T27 (16:00), 2017+'], thr['LGB T27 (16:00), 2017+'] = fit(C.T27, tr_17)
    preds['LGB T27+VIXPRE, 2017+'], thr['LGB T27+VIXPRE, 2017+'] = fit(C.T27 + C.VIXPRE, tr_17)
    preds['M1 rho x gap'], thr['M1 rho x gap'] = fit(['rho_gap'], tr_17, model='lr')
    print('base models done', flush=True)
    # D1: selection score (cross-sectional label) + combiner trained on out-of-sample selection scores
    sel_oos = np.clip(np.load(os.path.join(C.PRED, 'supp6_selection_oos_2017plus.npy')), 1e-6, 1 - 1e-6)
    z = np.log(sel_oos / (1 - sel_oos))
    sel_tr = z - pd.Series(z).groupby(d.to_numpy()).transform('mean').to_numpy()
    p_sel, _ = fit(C.T27, tr_all, label=ycs_tr)
    zh = np.log(np.clip(p_sel, 1e-6, 1 - 1e-6) / (1 - np.clip(p_sel, 1e-6, 1 - 1e-6)))
    sel_h = zh - pd.Series(zh).groupby(hold['Date'].to_numpy()).transform('mean').to_numpy()
    xt = np.column_stack([panel['rho_gap'].to_numpy(dtype='float64'), sel_tr]); xh = np.column_stack([hold['rho_gap'].to_numpy(dtype='float64'), sel_h])
    preds['D1 rho x gap + selection'], thr['D1 rho x gap + selection'] = fit(None, tr_17, model='lr', xh=xh, xt=xt)
    preds['selection score only (16:00)'] = sel_h
    kr_ok = panel[C.KRAUSS].notna().all(axis=1).to_numpy()
    p_kr, _ = fit(C.KRAUSS, tr_all & kr_ok, label=ycs_tr)
    p_kr = np.where(hold[C.KRAUSS].notna().all(axis=1).to_numpy(), p_kr, np.nan)
    g = hold['gap_z60'].to_numpy(dtype='float64')
    preds['zero-parameter: -VIX gap'] = -g
    preds['zero-parameter: -market return(t)'] = -hold['mkt_ret'].to_numpy(dtype='float64')
    np.save(os.path.join(C.PRED, 'holdout_preds.npy'), np.vstack([preds[k] for k in preds]))
    hold[['Ticker', 'Date', 'y', 'y_on', 'y_id', 'y_cs', 'fut_ret', 'on_ret', 'id_ret', 'gap_z60', 'rho', 'mkt_ret']].to_pickle(os.path.join(HOLD, 'holdout_frame.pkl'))

    boot = C.Boot(hold, n_boot=10000)
    days = pd.DatetimeIndex(boot.days)
    rows, cache = [], {}
    for nm, p in preds.items():
        r = {'model': nm}
        for lab in ('y', 'y_on', 'y_id'):
            pt, dr = boot.pooled(hold[lab].to_numpy(dtype='float64'), p)
            r[f'auroc_{lab}'] = pt; r[f'auroc_{lab}_lo'], r[f'auroc_{lab}_hi'] = ci(dr)
            cache[(nm, 'pooled', lab)] = (pt, dr)
        da = C.day_auc(hold, p)
        if da.notna().sum() > 0 and np.nanstd(da) > 0:
            pt, dr = boot.perday(da)
            r['perday'] = pt; r['perday_lo'], r['perday_hi'] = ci(dr)
            cache[(nm, 'perday', 'y')] = (pt, dr); cache[(nm, 'da')] = da
        if nm in thr:
            r['accuracy'] = float(((p > thr[nm]).astype(float) == hold['y'].to_numpy(dtype='float64')).mean())
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.RES, 'holdout_metrics.csv'), index=False)
    pairs = [('LGB T27 (16:00)', 'LGB T27+VIXPRE', 'pooled'), ('LGB T27 (16:00), 2017+', 'LGB T27+VIXPRE, 2017+', 'pooled'),
             ('LGB T27+VIXPRE, 2017+', 'M1 rho x gap', 'perday'), ('LGB T27+VIXPRE, 2017+', 'M1 rho x gap', 'pooled'),
             ('LGB T27+VIXPRE, 2017+', 'D1 rho x gap + selection', 'perday'), ('LGB T27+VIXPRE, 2017+', 'D1 rho x gap + selection', 'pooled'),
             ('M1 rho x gap', 'D1 rho x gap + selection', 'perday'), ('LGB T27+VIXPRE', 'M1 rho x gap', 'perday')]
    dl = pd.DataFrame([{'a': a, 'b': b, 'metric': met, **C.delta_stats(cache[(a, met, 'y')][1], cache[(b, met, 'y')][1], cache[(a, met, 'y')][0], cache[(b, met, 'y')][0])} for a, b, met in pairs])
    dl.to_csv(os.path.join(C.RES, 'holdout_deltas.csv'), index=False)

    # gap-size buckets for the cross-sectional ranking
    gd = hold.groupby('Date')['gap_z60'].first().reindex(days).to_numpy(dtype='float64')
    bk = []
    for nm in ('M1 rho x gap', 'D1 rho x gap + selection', 'LGB T27+VIXPRE, 2017+'):
        v = pd.Series(cache[(nm, 'da')]).reindex(days).to_numpy(dtype='float64')
        for lab, mk in (('|g|<0.5', np.abs(gd) < 0.5), ('0.5<=|g|<1.5', (np.abs(gd) >= 0.5) & (np.abs(gd) < 1.5)), ('|g|>=1.5', np.abs(gd) >= 1.5)):
            ok = (~np.isnan(v)) & mk
            den = boot.W @ ok.astype(np.float32)
            dr = np.where(den > 0, (boot.W @ np.where(ok, v, 0).astype(np.float32)) / np.maximum(den, 1e-9), np.nan)
            bk.append({'model': nm, 'bucket': lab, 'days': int(ok.sum()), 'perday_auc': float(v[ok].mean()), 'lo': ci(dr)[0], 'hi': ci(dr)[1]})
    bk = pd.DataFrame(bk)
    bk.to_csv(os.path.join(C.RES, 'holdout_buckets.csv'), index=False)

    # literature-style benchmark on the holdout: lagged-return features, beat-the-median label, k = 10
    k = 10
    dfx = pd.DataFrame({'Date': hold['Date'].to_numpy(), 'p': p_kr, 'ycs': hold['y_cs'].to_numpy(), 'r': hold['fut_ret'].to_numpy()}).dropna()
    dfx['hi'] = dfx.groupby('Date')['p'].rank(method='first', ascending=False); dfx['lo'] = dfx.groupby('Date')['p'].rank(method='first', ascending=True)
    top, flop = dfx[dfx['hi'] <= k], dfx[dfx['lo'] <= k]
    hit = (top.groupby('Date')['ycs'].sum().reindex(days).fillna(0) + (1 - flop['ycs']).groupby(flop['Date']).sum().reindex(days).fillna(0)).to_numpy(dtype=np.float32)
    cnt = (top.groupby('Date').size().reindex(days).fillna(0) + flop.groupby('Date').size().reindex(days).fillna(0)).to_numpy(dtype=np.float32)
    ls = (top.groupby('Date')['r'].mean() - flop.groupby('Date')['r'].mean()).reindex(days)
    okl = ls.notna().to_numpy().astype(np.float32); lsv = ls.fillna(0).to_numpy(dtype=np.float32)
    da_k = C.day_auc(hold, p_kr, 'y_cs')
    pk, dk = boot.perday(da_k)
    kr = {'krauss_perday_auc': pk, 'perday_lo': ci(dk)[0], 'perday_hi': ci(dk)[1], 'k10_accuracy': float(hit.sum() / cnt.sum()),
          'acc_lo': ci((boot.W @ hit) / (boot.W @ cnt))[0], 'acc_hi': ci((boot.W @ hit) / (boot.W @ cnt))[1],
          'k10_long_short_daily_return_pct': float(100 * lsv.sum() / okl.sum()), 'ls_lo': 100 * ci((boot.W @ lsv) / (boot.W @ okl))[0], 'ls_hi': 100 * ci((boot.W @ lsv) / (boot.W @ okl))[1]}

    # conversion law on the holdout
    th = load_module('10_supp_theory.py', 'supp_theory')
    dd = hold.groupby('Date').agg(g=('gap_z60', 'first'), M_cc=('fut_ret', 'mean'), M_on=('on_ret', 'mean'), M_id=('id_ret', 'mean')).reindex(days).reset_index(names='Date')
    bootd = C.Boot(dd, n_boot=10000)
    w_row = hold.groupby('Ticker').size()
    conv = []
    for lab, mc, yc, rc in (('close-to-close', 'M_cc', 'y', 'fut_ret'), ('overnight', 'M_on', 'y_on', 'on_ret'), ('intraday', 'M_id', 'y_id', 'id_ret')):
        s_day = -dd['g'].to_numpy(dtype='float64')
        a_m, dr_m = bootd.pooled(np.where(dd[mc].notna(), (dd[mc] > 0).astype(float), np.nan), s_day)
        a_s, dr_s = cache[('zero-parameter: -VIX gap', 'pooled', yc)]
        rho = th.static_rho(hold, rc)
        w = (w_row.reindex(rho.index) / w_row.reindex(rho.index).sum()).to_numpy()
        pred, pred_dr = float(th.predicted_stock_auc(a_m, rho.to_numpy(), w)), th.predicted_stock_auc(dr_m, rho.to_numpy(), w)
        conv.append({'label': lab, 'auc_market': a_m, 'mkt_lo': ci(dr_m)[0], 'mkt_hi': ci(dr_m)[1], 'rho_mean': float((rho.to_numpy() * w).sum()),
                     'auc_stock_predicted': pred, 'auc_stock_observed': a_s, 'obs_lo': ci(dr_s)[0], 'obs_hi': ci(dr_s)[1],
                     'obs_minus_pred': a_s - pred, 'diff_lo': ci(dr_s - pred_dr)[0], 'diff_hi': ci(dr_s - pred_dr)[1]})
    conv = pd.DataFrame(conv)
    conv.to_csv(os.path.join(C.RES, 'holdout_conversion.csv'), index=False)

    noazn = (hold['Ticker'] != 'AZN').to_numpy()
    summ = {'holdout_first_day': str(hold['Date'].min().date()), 'holdout_last_day': str(hold['Date'].max().date()), 'days': int(boot.D), 'bootstrap_blocks': int(np.ceil(boot.D / 20)),
            'rows': int(len(hold)), 'tickers': int(hold['Ticker'].nunique()), 'up_rate': float(hold['y'].mean()), 'train_up_rate': thr['LGB T27 (16:00)'],
            'market_return_sum_pct': float(100 * dd['M_cc'].sum()), 'days_gap_abs_ge_1p5': int((np.abs(gd) >= 1.5).sum()), 'train_last_day': str(d[tr_all].max().date()),
            'M1_perday_without_AZN': float(C.day_auc(hold[noazn].reset_index(drop=True), preds['M1 rho x gap'][noazn]).mean()),
            'M1_auroc_without_AZN': C._auc(hold['y'].to_numpy(dtype='float64')[noazn], preds['M1 rho x gap'][noazn]), **kr}
    json.dump(summ, open(os.path.join(C.RES, 'holdout_summary.json'), 'w'), indent=2)
    pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
    print(json.dumps(summ, indent=2))
    print(res.round(4).to_string(index=False))
    print(dl.round(4).to_string(index=False))
    print(bk.round(4).to_string(index=False))
    print(conv.round(4).to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'build':
        build()
    elif len(sys.argv) >= 2 and sys.argv[1] == 'eval':
        evaluate()
    else:
        sys.exit(__doc__)
