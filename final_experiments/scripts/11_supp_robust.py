"""Supp S8 - robustness and external validity with the data already in hand.
  main : (a) M1 by test year and its fitted coefficient per fold   (b) unseen stocks: fit on one half of the tickers, score the other half
         (c) entry cohorts: tickers with a long history vs late entrants
  era  : (d) 16:00-information models walked forward over 2005-2026, summarised by era (the VIX gap is not available before 2017)
Usage: python 11_supp_robust.py main | era
Limitation that no check here removes: the universe is today's NASDAQ-100 members, so earlier years hold survivors only.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def ci(draws):
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def masked_perday(boot, tf, score, row_mask, day_mask=None):
    """Mean per-day AUC of `score` restricted to rows in row_mask (ranking within that subset) and days in day_mask."""
    da = C.day_auc(tf, np.where(row_mask, score, np.nan))
    v = pd.Series(da).reindex(pd.DatetimeIndex(boot.days)).to_numpy(dtype='float64')
    ok = ~np.isnan(v) if day_mask is None else (~np.isnan(v)) & day_mask
    a = np.where(ok, v, 0).astype(np.float32)
    den = boot.W @ ok.astype(np.float32)
    return float(v[ok].mean()), np.where(den > 0, (boot.W @ a) / np.maximum(den, 1e-9), np.nan), int(ok.sum())


def masked_pooled(boot, tf, score, row_mask, label='y'):
    y = np.where(row_mask, tf[label].to_numpy(dtype='float64'), np.nan)
    return boot.pooled(y, score)


def walk_2017(panel, feats, model, fit_rows):
    """Walk-forward on 2017+ rows, trained only on rows in fit_rows; predictions for every test row."""
    x = panel[feats].to_numpy(dtype='float64')
    y_all = panel['y'].to_numpy(dtype='float64')
    is_test = (panel['Date'] >= '2018-01-01').to_numpy()
    p = np.full(len(panel), np.nan)
    for yr, tr, te in C.folds(panel, '2017-01-01'):
        ok = tr & fit_rows
        if model == 'lr':
            ok &= ~np.isnan(x).any(axis=1)
        p[te] = C.fit_predict(x[ok], y_all[ok], x[te], model)
    return p[is_test]


def main():
    panel = C.load_panel()
    panel['rho_gap'] = panel['rho'] * panel['gap_z60']
    tf = C.test_frame(panel)
    boot = C.Boot(tf, n_boot=10000)
    days = pd.DatetimeIndex(boot.days)
    m1 = C.load_pred('supp6_D0_rho_gap')[0]
    ref = C.load_pred('lgb_T27+VIXPRE_common')[0]
    allrows = np.ones(len(tf), bool)
    y = tf['y'].to_numpy(dtype='float64')
    yrs_row = tf['Date'].dt.year.to_numpy()

    # (a) by year
    by = []
    x = panel[['rho_gap']].to_numpy(dtype='float64'); y_all = panel['y'].to_numpy(dtype='float64')
    coef = {}
    for yr, tr, te in C.folds(panel, '2017-01-01'):
        ok = tr & ~np.isnan(x).any(axis=1)
        m = LogisticRegression(C=1e6, max_iter=2000).fit(x[ok], y_all[ok])
        coef[yr] = (float(m.intercept_[0]), float(m.coef_[0][0]), int(ok.sum()))
    for yr in C.TEST_YEARS:
        dm = np.asarray(days.year == yr)
        pt, dr, n = masked_perday(boot, tf, m1, allrows, dm)
        pr, drr, _ = masked_perday(boot, tf, ref, allrows, dm)
        k = yrs_row == yr
        by.append({'year': yr, 'days': n, 'M1_perday': pt, 'lo': ci(dr)[0], 'hi': ci(dr)[1], 'LGB_perday': pr, 'M1_minus_LGB': pt - pr,
                   'diff_lo': ci(dr - drr)[0], 'diff_hi': ci(dr - drr)[1], 'M1_auroc': C._auc(y[k], m1[k]), 'LGB_auroc': C._auc(y[k], ref[k]),
                   'fit_intercept': coef[yr][0], 'fit_b_rho_gap': coef[yr][1], 'train_rows': coef[yr][2]})
    by = pd.DataFrame(by)
    by.to_csv(os.path.join(C.RES, 'supp8_by_year.csv'), index=False)

    # (b) unseen stocks
    tick_panel = panel['Ticker'].to_numpy()
    tick_tf = tf['Ticker'].to_numpy()
    tickers = np.sort(panel['Ticker'].unique())
    feats_lgb = C.T27 + C.VIXPRE
    rows = []
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        half_a = set(rng.permutation(tickers)[: len(tickers) // 2])
        in_a_panel = np.isin(tick_panel, list(half_a)); in_a_tf = np.isin(tick_tf, list(half_a))
        jobs = [('M1 rho x gap', ['rho_gap'], 'lr', m1)]
        if seed < 3:
            jobs.append(('LightGBM T27+VIXPRE', feats_lgb, 'lgb', ref))
        for name, feats, model, seen in jobs:
            p_from_a = walk_2017(panel, feats, model, in_a_panel)        # trained on half A
            p_from_b = walk_2017(panel, feats, model, ~in_a_panel)       # trained on half B
            for metric in ('perday', 'pooled'):
                vals = {}
                for kind, pa, pb in (('unseen', p_from_b, p_from_a), ('seen', seen, seen)):
                    if metric == 'perday':
                        a_pt, a_dr, _ = masked_perday(boot, tf, pa, in_a_tf)      # rows of half A, scored by the model that never saw them
                        b_pt, b_dr, _ = masked_perday(boot, tf, pb, ~in_a_tf)
                    else:
                        a_pt, a_dr = masked_pooled(boot, tf, pa, in_a_tf)
                        b_pt, b_dr = masked_pooled(boot, tf, pb, ~in_a_tf)
                    vals[kind] = ((a_pt + b_pt) / 2, (a_dr + b_dr) / 2)
                d = vals['unseen'][1] - vals['seen'][1]
                rows.append({'model': name, 'seed': seed, 'metric': metric, 'seen': vals['seen'][0], 'unseen': vals['unseen'][0],
                             'unseen_lo': ci(vals['unseen'][1])[0], 'unseen_hi': ci(vals['unseen'][1])[1],
                             'unseen_minus_seen': vals['unseen'][0] - vals['seen'][0], 'diff_lo': ci(d)[0], 'diff_hi': ci(d)[1]})
        print('split', seed, 'done', flush=True)
    un = pd.DataFrame(rows)
    un.to_csv(os.path.join(C.RES, 'supp8_unseen_stocks.csv'), index=False)

    # (c) entry cohorts
    first = panel.groupby('Ticker')['Date'].min()
    cohorts = {'in the panel since 2002-2003': first[first <= '2003-12-31'].index, 'entered 2004-2009': first[(first > '2003-12-31') & (first < '2010-01-01')].index,
               'entered 2010 or later': first[first >= '2010-01-01'].index}
    co = []
    for nm, tk in cohorts.items():
        rm = np.isin(tick_tf, list(tk))
        pt, dr, n = masked_perday(boot, tf, m1, rm)
        pp, drp = masked_pooled(boot, tf, m1, rm)
        pr, drr, _ = masked_perday(boot, tf, ref, rm)
        co.append({'cohort': nm, 'tickers': len(tk), 'test_rows': int(rm.sum()), 'M1_perday': pt, 'lo': ci(dr)[0], 'hi': ci(dr)[1],
                   'M1_auroc': pp, 'auroc_lo': ci(drp)[0], 'auroc_hi': ci(drp)[1], 'LGB_perday': pr,
                   'rho_mean': float(tf.loc[rm, 'rho'].mean())})
    co = pd.DataFrame(co)
    co.to_csv(os.path.join(C.RES, 'supp8_cohorts.csv'), index=False)

    pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
    print(by.round(4).to_string(index=False))
    print(un.round(4).to_string(index=False))
    print(un.groupby(['model', 'metric'])[['seen', 'unseen', 'unseen_minus_seen']].mean().round(4).to_string())
    print(co.round(4).to_string(index=False))


def era():
    panel = C.load_panel()
    med = panel.groupby('Date')['fut_ret'].transform('median')
    panel['y_cs'] = (panel['fut_ret'] > med).astype('float64')
    d = panel['Date']
    years = range(2005, 2027)

    def walk(feats, label, need):
        x = panel[feats].to_numpy(dtype='float64'); yv = panel[label].to_numpy(dtype='float64')
        okrow = panel[need].notna().all(axis=1).to_numpy() if need else np.ones(len(panel), bool)
        p = np.full(len(panel), np.nan)
        for yr in years:
            te = (d.dt.year == yr).to_numpy() & okrow
            tr = d < pd.Timestamp(yr, 1, 1)
            tr &= d < d[tr].max()
            tr = tr.to_numpy() & okrow
            if tr.sum() < 5000 or not te.any():
                continue
            p[te] = C.fit_predict(x[tr], yv[tr], x[te], 'lgb')
        return p

    cache = os.path.join(C.PRED, 'supp8_era_preds.npy')
    if os.path.exists(cache):
        pk_all, pt_all = np.load(cache)
    else:
        pk_all = walk(C.KRAUSS, 'y_cs', C.KRAUSS); print('krauss walk done', flush=True)
        pt_all = walk(C.T27, 'y', None); print('T27 walk done', flush=True)
        np.save(cache, np.vstack([pk_all, pt_all]))
    keep = (d >= '2005-01-01').to_numpy()
    tfe = panel.loc[keep, ['Ticker', 'Date', 'y', 'y_cs', 'fut_ret', 'mkt_ret']].reset_index(drop=True)
    pk, pt = pk_all[keep], pt_all[keep]
    boot = C.Boot(tfe, n_boot=5000)
    days = pd.DatetimeIndex(boot.days)
    y, ycs = tfe['y'].to_numpy(dtype='float64'), tfe['y_cs'].to_numpy(dtype='float64')
    yr_row = tfe['Date'].dt.year.to_numpy()
    rev = -tfe['mkt_ret'].to_numpy(dtype='float64')
    da_k = pd.Series(C.day_auc(tfe, pk, 'y_cs')).reindex(days).to_numpy(dtype='float64')
    da_t = pd.Series(C.day_auc(tfe, pt, 'y')).reindex(days).to_numpy(dtype='float64')

    k = 10
    dfx = pd.DataFrame({'Date': tfe['Date'].to_numpy(), 'p': pk, 'ycs': ycs}).dropna()
    dfx = dfx[dfx.groupby('Date')['p'].transform('size') >= 2 * k]
    dfx['hi'] = dfx.groupby('Date')['p'].rank(method='first', ascending=False)
    dfx['lo'] = dfx.groupby('Date')['p'].rank(method='first', ascending=True)
    top, flop = dfx[dfx['hi'] <= k], dfx[dfx['lo'] <= k]
    hit = (top.groupby('Date')['ycs'].sum().reindex(days).fillna(0) + (1 - flop['ycs']).groupby(flop['Date']).sum().reindex(days).fillna(0)).to_numpy(dtype=np.float32)
    cnt = (top.groupby('Date').size().reindex(days).fillna(0) + flop.groupby('Date').size().reindex(days).fillna(0)).to_numpy(dtype=np.float32)
    n_day = tfe.groupby('Date').size().reindex(days).to_numpy(dtype='float64')

    def dmean(v, dm):
        ok = (~np.isnan(v)) & dm
        a = np.where(ok, v, 0).astype(np.float32)
        return float(v[ok].mean()), (boot.W @ a) / np.maximum(boot.W @ ok.astype(np.float32), 1e-9)

    eras = {'2005-2009': (2005, 2009), '2010-2017': (2010, 2017), '2018-2026': (2018, 2026)}
    out, keepdraw = [], {}
    for nm, (a, b) in eras.items():
        dm = np.asarray((days.year >= a) & (days.year <= b))
        rm = (yr_row >= a) & (yr_row <= b)
        r = {'era': nm, 'days': int(dm.sum()), 'stocks_per_day': float(np.nanmean(n_day[dm]))}
        v, dr = dmean(da_k, dm); r['krauss_cs_perday'] = v; r['k_lo'], r['k_hi'] = ci(dr); keepdraw[(nm, 'k')] = (v, dr)
        acc = float(hit[dm].sum() / cnt[dm].sum()); dra = (boot.W @ (hit * dm)) / np.maximum(boot.W @ (cnt * dm), 1e-9)
        r['krauss_k10_accuracy'] = acc; r['acc_lo'], r['acc_hi'] = ci(dra); keepdraw[(nm, 'acc')] = (acc, dra)
        v, dr = dmean(da_t, dm); r['T27_perday'] = v; r['t_lo'], r['t_hi'] = ci(dr)
        v, dr = boot.pooled(np.where(rm, y, np.nan), pt); r['T27_auroc'] = v; r['ta_lo'], r['ta_hi'] = ci(dr); keepdraw[(nm, 't')] = (v, dr)
        v, dr = boot.pooled(np.where(rm, y, np.nan), rev); r['mkt_reversal_auroc'] = v; r['rev_lo'], r['rev_hi'] = ci(dr); keepdraw[(nm, 'rev')] = (v, dr)
        r['up_rate'] = float(np.nanmean(y[rm]))
        out.append(r)
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(C.RES, 'supp8_era.csv'), index=False)
    dl = []
    for key, lab in (('k', 'Krauss features, per-day AUC'), ('acc', 'Krauss features, k=10 accuracy'), ('t', 'T27 pooled AUROC'), ('rev', '-market return(t) pooled AUROC')):
        for early in ('2005-2009', '2010-2017'):
            A, B = keepdraw[(early, key)], keepdraw[('2018-2026', key)]
            dl.append({'metric': lab, 'from': early, 'to': '2018-2026', **C.delta_stats(A[1], B[1], A[0], B[0])})
    dl = pd.DataFrame(dl)
    dl.to_csv(os.path.join(C.RES, 'supp8_era_deltas.csv'), index=False)
    yearly = pd.DataFrame({'year': list(years)})
    yearly['krauss_cs_perday'] = [float(np.nanmean(da_k[np.asarray(days.year == yv)])) for yv in years]
    yearly['T27_auroc'] = [C._auc(y[yr_row == yv], pt[yr_row == yv]) for yv in years]
    yearly['mkt_reversal_auroc'] = [C._auc(y[yr_row == yv], rev[yr_row == yv]) for yv in years]
    yearly['stocks_per_day'] = [float(np.nanmean(n_day[np.asarray(days.year == yv)])) for yv in years]
    yearly.to_csv(os.path.join(C.RES, 'supp8_era_by_year.csv'), index=False)
    pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
    print(out.round(4).to_string(index=False))
    print(dl.round(4).to_string(index=False))
    print(yearly.round(4).to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'main':
        main()
    elif len(sys.argv) >= 2 and sys.argv[1] == 'era':
        era()
    else:
        sys.exit(__doc__)
