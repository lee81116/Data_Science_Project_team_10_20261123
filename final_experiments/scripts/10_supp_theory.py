"""Supp S7 - tests of the one-factor structure behind M1  (r_i = beta_i * m + eps_i).
(a) exposure definition: rho vs beta vs total volatility, plus a within-day shuffle placebo
(b) conversion law: the stock-level AUROC of a market-level signal is its market-level AUROC diluted by rho
        AUC_stock = 1/2 + (2/pi) * mean_i asin(kappa * rho_i / sqrt(2)),   kappa = sqrt(2) * sin(pi * (AUC_mkt - 1/2) / 2)
(c) model-implied per-day AUC of the rho ranking, given the realised market move:
        P(y_i = 1 | z) = Phi((Phi^-1(p_bar) + rho_i * z) / sqrt(1 - rho_i^2))      (one-factor conditional probability, Vasicek 2002)
(d) calibration of M1 / D1 against the monolithic LightGBM
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ROLL = dict(window=252, min_periods=126)


def rolling_exposures(panel):
    wide = panel.pivot(index='Date', columns='Ticker', values='return_1d').astype('float64')
    n = wide.notna().sum(axis=1)
    tot = wide.sum(axis=1, skipna=True)
    out = {}
    for c in wide.columns:
        r = wide[c]
        others = n - r.notna().astype(int)
        ex = ((tot - r.fillna(0)) / others.where(others > 0)).where(r.notna())      # leave-one-out market return
        out[c] = pd.DataFrame({'rho2': r.rolling(**ROLL).corr(ex).clip(-1, 1), 'beta': r.rolling(**ROLL).cov(ex) / ex.rolling(**ROLL).var(),
                               'sig': r.rolling(**ROLL).std()})
    return pd.concat(out, names=['Ticker', 'Date']).reset_index()


def ci(draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def static_rho(tf, col):
    """Test-period correlation of each stock's return with the leave-one-out equal-weighted market return (same interval)."""
    wide = tf.pivot(index='Date', columns='Ticker', values=col).astype('float64')
    n = wide.notna().sum(axis=1)
    tot = wide.sum(axis=1, skipna=True)
    ex = (-wide.fillna(0)).add(tot, axis=0).div((n - 1).where(n > 1), axis=0).where(wide.notna())
    rho = wide.corrwith(ex)
    return rho[wide.notna().sum() >= 60]


def predicted_stock_auc(auc_mkt, rho, w):
    kappa = np.sqrt(2) * np.sin(np.pi * (np.asarray(auc_mkt) - 0.5) / 2)
    kr = np.clip(np.multiply.outer(kappa, rho) / np.sqrt(2), -1, 1)
    return 0.5 + (2 / np.pi) * (np.arcsin(kr) * w).sum(axis=-1)


def implied_auc_day(p, s):
    k = (s[:, None] > s[None, :]).astype('float64') + 0.5 * (s[:, None] == s[None, :])
    w = p[:, None] * (1 - p)[None, :]
    np.fill_diagonal(w, 0.0)
    return float((w * k).sum() / w.sum())


def main():
    panel = C.load_panel()
    panel = panel.merge(rolling_exposures(panel), on=['Ticker', 'Date'], how='left')
    is_test = (panel['Date'] >= '2018-01-01').to_numpy()
    tf = panel.loc[is_test, ['Ticker', 'Date', 'y', 'y_on', 'y_id', 'fut_ret', 'on_ret', 'id_ret', 'gap_z60', 'gap_lag_z60', 'gap_shuf_z60',
                             'rho', 'rho2', 'beta', 'sig', 'mkt_ret']].reset_index(drop=True)
    boot = C.Boot(tf, n_boot=10000)
    days = pd.DatetimeIndex(boot.days)
    g = tf['gap_z60'].to_numpy(dtype='float64')
    chk = tf[['rho', 'rho2']].dropna()
    print('stored rho vs recomputed rho2: corr', round(float(chk.corr().iloc[0, 1]), 5), '| mean abs diff', round(float((chk.rho - chk.rho2).abs().mean()), 5))

    # ---------------------------------------------------------------- (a) exposure definition + placebo
    scores = {'rho (stored)': -tf['rho'].to_numpy(dtype='float64') * g, 'rho (recomputed, same windows as beta)': -tf['rho2'].to_numpy(dtype='float64') * g,
              'beta': -tf['beta'].to_numpy(dtype='float64') * g, 'total volatility': -tf['sig'].to_numpy(dtype='float64') * g}
    rows, cache = [], {}
    for nm, s in scores.items():
        pt, dr = boot.perday(C.day_auc(tf, s))
        cache[nm] = (pt, dr)
        rows.append({'exposure': nm, 'perday_auc': pt, 'lo': ci(dr)[0], 'hi': ci(dr)[1]})
    base = 'rho (recomputed, same windows as beta)'
    for nm in ('beta', 'total volatility'):
        d = C.delta_stats(cache[nm][1], cache[base][1], cache[nm][0], cache[base][0])
        rows.append({'exposure': f'rho minus {nm}', 'perday_auc': d['delta'], 'lo': d['lo'], 'hi': d['hi'], 'p': d['p']})
    ok = ~(np.isnan(tf['rho'].to_numpy(dtype='float64')) | np.isnan(g))
    day_idx = boot.day_idx[ok]
    by_day = np.argsort(day_idx, kind='stable')
    rho_ok = tf['rho'].to_numpy(dtype='float64')[ok]
    plc = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        perm = np.lexsort((rng.random(ok.sum()), day_idx))
        sh = np.empty(ok.sum()); sh[perm] = rho_ok[by_day]
        s = np.full(len(tf), np.nan); s[ok] = -sh * g[ok]
        plc.append(float(C.day_auc(tf, s).mean()))
    rows.append({'exposure': 'placebo: rho shuffled across stocks within the day (20 seeds, mean)', 'perday_auc': float(np.mean(plc)),
                 'lo': float(np.min(plc)), 'hi': float(np.max(plc))})
    exp = pd.DataFrame(rows)
    exp.to_csv(os.path.join(C.RES, 'supp7_exposure.csv'), index=False)
    cs = tf.groupby('Date')[['rho', 'beta', 'sig']].corr(method='spearman')
    print('avg within-day Spearman: rho~beta', round(float(cs.xs('rho', level=1)['beta'].mean()), 3), '| rho~sig', round(float(cs.xs('rho', level=1)['sig'].mean()), 3),
          '| beta~sig', round(float(cs.xs('beta', level=1)['sig'].mean()), 3))

    # ---------------------------------------------------------------- (b) conversion law
    dd = tf.groupby('Date').agg(g=('gap_z60', 'first'), g_lag=('gap_lag_z60', 'first'), g_shuf=('gap_shuf_z60', 'first'), m_t=('mkt_ret', 'first'),
                                M_cc=('fut_ret', 'mean'), M_on=('on_ret', 'mean'), M_id=('id_ret', 'mean')).reindex(days).reset_index(names='Date')
    bootd = C.Boot(dd, n_boot=10000)
    assert np.array_equal(bootd.W, boot.W), 'day-level and stock-level bootstrap weights differ'
    w_row = tf.groupby('Ticker').size()
    signals = {'-VIX gap (03:15)': ('g', -1), '-market return of day t (16:00)': ('m_t', -1), '-lagged VIX gap (16:00, placebo)': ('g_lag', -1),
               '-shuffled VIX gap (placebo)': ('g_shuf', -1), 'oracle: realised market return': (None, 1)}
    labels = {'close-to-close': ('M_cc', 'y', 'fut_ret'), 'overnight': ('M_on', 'y_on', 'on_ret'), 'intraday': ('M_id', 'y_id', 'id_ret')}
    rhos = {lab: static_rho(tf, rc) for lab, (_, _, rc) in labels.items()}
    conv = []
    for sn, (col, sign) in signals.items():
        for lab, (mc, yc, rc) in labels.items():
            if sn.startswith('-market') and lab != 'close-to-close':
                continue
            s_day = sign * (dd[col] if col else dd[mc]).to_numpy(dtype='float64')
            y_mkt = np.where(dd[mc].notna(), (dd[mc] > 0).astype(float), np.nan)
            a_m, dr_m = bootd.pooled(y_mkt, s_day)
            s_row = pd.Series(s_day, index=days).reindex(tf['Date']).to_numpy()
            a_s, dr_s = boot.pooled(tf[yc].to_numpy(dtype='float64'), s_row)
            rho = rhos[lab]
            w = (w_row.reindex(rho.index) / w_row.reindex(rho.index).sum()).to_numpy()
            pred, pred_dr = float(predicted_stock_auc(a_m, rho.to_numpy(), w)), predicted_stock_auc(dr_m, rho.to_numpy(), w)
            gap = dr_s - pred_dr
            conv.append({'signal': sn, 'label': lab, 'auc_market': a_m, 'auc_market_lo': ci(dr_m)[0], 'auc_market_hi': ci(dr_m)[1],
                         'rho_mean': float((rho.to_numpy() * w).sum()), 'auc_stock_predicted': pred, 'pred_lo': ci(pred_dr)[0], 'pred_hi': ci(pred_dr)[1],
                         'auc_stock_first_order': 0.5 + float((rho.to_numpy() * w).sum()) * (a_m - 0.5),
                         'auc_stock_observed': a_s, 'obs_lo': ci(dr_s)[0], 'obs_hi': ci(dr_s)[1],
                         'obs_minus_pred': a_s - pred, 'diff_lo': ci(gap)[0], 'diff_hi': ci(gap)[1]})
    conv = pd.DataFrame(conv)
    conv.to_csv(os.path.join(C.RES, 'supp7_conversion.csv'), index=False)
    rho_cc = rhos['close-to-close']
    w_cc = (w_row.reindex(rho_cc.index) / w_row.reindex(rho_cc.index).sum()).to_numpy()
    grid = np.linspace(0.5, 1.0, 50001)
    pred_grid = predicted_stock_auc(grid, rho_cc.to_numpy(), w_cc)
    req = [{'target_stock_auroc': t, 'required_market_auc': float(grid[np.searchsorted(pred_grid, t)]) if t <= pred_grid[-1] else np.nan} for t in (0.51, 0.52, 0.55, 0.60, 0.65, 0.70)]
    req.append({'target_stock_auroc': float(pred_grid[-1]), 'required_market_auc': 1.0})
    req = pd.DataFrame(req)
    req.to_csv(os.path.join(C.RES, 'supp7_required_skill.csv'), index=False)

    # ---------------------------------------------------------------- (c) model-implied per-day AUC
    m_all = panel.groupby('Date')['mkt_ret'].first()
    sig_m = m_all.rolling(**ROLL).std().reindex(days).to_numpy()
    p_bar = float(panel.loc[panel['Date'] < '2018-01-01', 'y'].mean())
    c0 = norm.ppf(p_bar)
    z = dd['M_cc'].to_numpy(dtype='float64') / sig_m
    gd = dd['g'].to_numpy(dtype='float64')
    imp, imp_or = np.full(len(days), np.nan), np.full(len(days), np.nan)
    grp = {d: k for d, k in tf.groupby('Date').indices.items()}
    rho_all = tf['rho'].to_numpy(dtype='float64')
    for i, d in enumerate(days):
        k = grp.get(d)
        if k is None or np.isnan(z[i]):
            continue
        r = rho_all[k]; r = np.clip(r[~np.isnan(r)], -0.99, 0.99)
        if len(r) < 10:
            continue
        p = norm.cdf((c0 + r * z[i]) / np.sqrt(1 - r ** 2))
        if not np.isnan(gd[i]) and gd[i] != 0:
            imp[i] = implied_auc_day(p, -r * np.sign(gd[i]))
        if z[i] != 0:
            imp_or[i] = implied_auc_day(p, r * np.sign(z[i]))
    obs = C.day_auc(tf, -rho_all * g).reindex(days).to_numpy(dtype='float64')
    m_row = pd.Series(dd['M_cc'].to_numpy(), index=days).reindex(tf['Date']).to_numpy(dtype='float64')
    obs_or = C.day_auc(tf, rho_all * np.sign(m_row)).reindex(days).to_numpy(dtype='float64')
    buckets = {'all days': np.ones(len(days), bool), '|g|<0.5': np.abs(gd) < 0.5, '0.5<=|g|<1.5': (np.abs(gd) >= 0.5) & (np.abs(gd) < 1.5), '|g|>=1.5': np.abs(gd) >= 1.5,
               'g<=-1.5': gd <= -1.5, 'g<0': gd < 0, 'g>0': gd > 0, 'g>=+1.5': gd >= 1.5}

    def bstat(v, mk):
        okk = (~np.isnan(v)) & mk
        a = np.where(okk, v, 0).astype(np.float32)
        return float(v[okk].mean()), (boot.W @ a) / np.maximum(boot.W @ okk.astype(np.float32), 1e-9), int(okk.sum())

    ip = []
    for direction, o, m in (('direction from the VIX gap sign', obs, imp), ('oracle direction (realised market sign)', obs_or, imp_or)):
        for lab, mk in buckets.items():
            if direction.startswith('oracle') and lab != 'all days':
                continue
            both = mk & ~np.isnan(o) & ~np.isnan(m)
            po, do, n = bstat(o, both)
            pm, dm, _ = bstat(m, both)
            ip.append({'direction': direction, 'bucket': lab, 'days': n, 'observed': po, 'obs_lo': ci(do)[0], 'obs_hi': ci(do)[1],
                       'implied': pm, 'imp_lo': ci(dm)[0], 'imp_hi': ci(dm)[1], 'obs_minus_implied': po - pm, 'diff_lo': ci(do - dm)[0], 'diff_hi': ci(do - dm)[1]})
    ip = pd.DataFrame(ip)
    ip.to_csv(os.path.join(C.RES, 'supp7_implied_perday.csv'), index=False)
    okd = ~np.isnan(obs) & ~np.isnan(imp)
    print('day-level correlation of observed and implied per-day AUC:', round(float(np.corrcoef(obs[okd], imp[okd])[0, 1]), 3), '| p_bar', round(p_bar, 4))

    # ---------------------------------------------------------------- (d) calibration
    tfm = C.test_frame(panel)
    assert len(tfm) == len(tf)
    y = tfm['y'].to_numpy(dtype='float64')
    models = {'M1 rho x gap (2 parameters)': C.load_pred('supp6_D0_rho_gap'), 'D1 rho x gap + selection (3 parameters)': C.load_pred('supp6_D1_rho_gap+sel'),
              'LightGBM T27+VIXPRE (2017+)': C.load_pred('lgb_T27+VIXPRE_common')}
    models['training base rate only'] = (models['M1 rho x gap (2 parameters)'][1], None)
    cal, bins = [], []
    sq = {}
    for nm, (p, _) in models.items():
        p = np.clip(p, 1e-6, 1 - 1e-6)
        sq[nm] = (p - y) ** 2
        q = pd.qcut(p, 10, labels=False, duplicates='drop')
        b = pd.DataFrame({'bin': q, 'p': p, 'y': y}).groupby('bin').agg(mean_pred=('p', 'mean'), up_rate=('y', 'mean'), n=('y', 'size')).reset_index()
        b.insert(0, 'model', nm)
        bins.append(b)
        cal.append({'model': nm, 'brier': float(sq[nm].mean()), 'log_loss': float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                    'ece_10bins': float((b['n'] * (b['mean_pred'] - b['up_rate']).abs()).sum() / b['n'].sum()), 'pred_sd': float(p.std())})
    base_day = np.bincount(boot.day_idx, weights=sq['training base rate only'], minlength=boot.D).astype(np.float32)
    for r in cal:
        m_day = np.bincount(boot.day_idx, weights=sq[r['model']], minlength=boot.D).astype(np.float32)
        bss = 1 - (boot.W @ m_day) / (boot.W @ base_day)
        r['brier_skill_vs_base_rate'] = 1 - float(m_day.sum() / base_day.sum()); r['bss_lo'], r['bss_hi'] = ci(bss)
    cal = pd.DataFrame(cal)
    cal.to_csv(os.path.join(C.RES, 'supp7_calibration.csv'), index=False)
    n_day = np.bincount(boot.day_idx, minlength=boot.D).astype(np.float32)
    a_day = np.bincount(boot.day_idx, weights=sq['LightGBM T27+VIXPRE (2017+)'], minlength=boot.D).astype(np.float32)
    b_day = np.bincount(boot.day_idx, weights=sq['M1 rho x gap (2 parameters)'], minlength=boot.D).astype(np.float32)
    dd_b = (boot.W @ a_day) / (boot.W @ n_day) - (boot.W @ b_day) / (boot.W @ n_day)
    import json
    json.dump({'brier_lgb_minus_m1': float(a_day.sum() / n_day.sum() - b_day.sum() / n_day.sum()), 'lo': ci(dd_b)[0], 'hi': ci(dd_b)[1],
               'p_two_sided': float(2 * min((dd_b <= 0).mean(), (dd_b >= 0).mean()))}, open(os.path.join(C.RES, 'supp7_brier_paired.json'), 'w'), indent=2)
    pd.concat(bins).to_csv(os.path.join(C.RES, 'supp7_calibration_bins.csv'), index=False)

    pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
    print(exp.round(4).to_string(index=False))
    print(conv.round(4).to_string(index=False))
    print(req.round(4).to_string(index=False))
    print(ip.round(4).to_string(index=False))
    print(cal.round(5).to_string(index=False))
    print(pd.concat(bins).query("model != 'training base rate only'").round(4).to_string(index=False))


if __name__ == '__main__':
    main()
