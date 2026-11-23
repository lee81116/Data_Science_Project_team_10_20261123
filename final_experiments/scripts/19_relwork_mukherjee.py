"""Baseline of Mukherjee et al. (2025, HICSS-58, "Beyond the bell") re-run on our panel.

Their baseline: 15 technical indicators of the previous day (their Table 2) -> XGBoost regressor of the end-of-day
price -> direction = predicted price vs previous end-of-day price -> balanced accuracy per ticker, averaged.
They report 56.52% on 473 S&P 500 stocks, 300 test days.

What the paper does not state, and what is assumed here:
  * one model per ticker (a pooled regression of price levels across tickers is not meaningful);
  * XGBoost default hyper-parameters (none are reported);
  * training on all earlier days of the ticker;
  * Schaff Trend Cycle: the printed formula is not well defined, so the standard definition is used with the
    windows of their Table 2 (MACD 12/26, 14-day stochastic, 3-day smoothing).
On our side: prices are dividend/split adjusted, rows are the analysis panel (2002+, Adj Close >= $5), label y = 1 if
AdjClose(t+1) > AdjClose(t). In our row convention the features of day t predict the price of day t+1.

Usage:
  python3 19_relwork_mukherjee.py ind                 # indicators -> data/mukherjee_ind.pkl
  python3 19_relwork_mukherjee.py fitA [target]       # protocol A: last 300 rows per ticker (as in the paper)
  python3 19_relwork_mukherjee.py fitB 2018 2019 ...  # protocol B: our yearly walk-forward, cached per year
  python3 19_relwork_mukherjee.py eval
target: 'price' (paper; default) or 'ret' (same features, next-day return as the regression target; a check of ours).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

RAW = os.path.join(C.ROOT, '..', 'dataset', 'archive', 'NASDAQ100_Historical_Data.csv')
IND = os.path.join(C.DATA, 'mukherjee_ind.pkl')
CACHE = os.path.join(C.PRED, 'mukherjee')
FEATS = ['vwap', 'adi', 'fi', 'mfi', 'eom', 'ppo', 'roc', 'rsi', 'tsi', 'willr', 'mi', 'stc', 'trix', 'dpo', 'cci']
N_TEST, MIN_TRAIN = 300, 300
XGB_PARAMS = dict(n_estimators=100, n_jobs=4, random_state=42)          # library defaults otherwise


def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def one_stock(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values('Date').copy()
    c, h, l, v = g['P'], g['H'], g['L'], g['V']
    tp = (h + l + c) / 3
    rng = (h - l).where((h - l) > 0)
    pc = c.diff()
    out = pd.DataFrame({'Ticker': g['Ticker'], 'Date': g['Date']})
    out['vwap'] = (v * tp).rolling(14, min_periods=14).sum() / v.rolling(14, min_periods=14).sum()
    out['adi'] = (v * (((c - l) - (h - c)) / rng)).fillna(0).cumsum()
    out['fi'] = ema(pc * v, 13)
    rmf = v * tp
    up = rmf.where(tp.diff() > 0, 0.0).rolling(14, min_periods=14).sum()
    dn = rmf.where(tp.diff() < 0, 0.0).rolling(14, min_periods=14).sum()
    out['mfi'] = 100 - 100 / (1 + up / dn.where(dn > 0))
    dm = (h + l) / 2 - (h.shift(1) + l.shift(1)) / 2
    out['eom'] = (dm / ((v / 1e8) / rng)).rolling(14, min_periods=14).mean()
    e12, e26 = ema(c, 12), ema(c, 26)
    out['ppo'] = 100 * (e12 - e26) / e26
    out['roc'] = 100 * (c / c.shift(12) - 1)
    ag = pc.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    al = (-pc.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out['rsi'] = 100 - 100 / (1 + ag / al.where(al > 0))
    out['tsi'] = 100 * ema(ema(pc, 25), 13) / ema(ema(pc.abs(), 25), 13)
    hh, ll = h.rolling(14, min_periods=14).max(), l.rolling(14, min_periods=14).min()
    out['willr'] = -100 * (hh - c) / (hh - ll).where((hh - ll) > 0)
    e9 = ema(h - l, 9)
    out['mi'] = (e9 / ema(e9, 9)).rolling(25, min_periods=25).sum()
    macd = e12 - e26

    def stoch(x):
        lo, hi = x.rolling(14, min_periods=14).min(), x.rolling(14, min_periods=14).max()
        return 100 * (x - lo) / (hi - lo).where((hi - lo) > 0)

    d1 = stoch(macd).rolling(3, min_periods=3).mean()
    out['stc'] = stoch(d1).rolling(3, min_periods=3).mean()
    t3 = ema(ema(ema(c, 15), 15), 15)
    out['trix'] = 100 * (t3 / t3.shift(1) - 1)
    sma20 = c.rolling(20, min_periods=20).mean()
    out['dpo'] = c.shift(11) - sma20
    stp = tp.rolling(20, min_periods=20).mean()
    md = (tp - stp).abs().rolling(20, min_periods=20).mean()          # mean deviation around the current SMA is approximated per day
    md = tp.rolling(20, min_periods=20).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True)
    out['cci'] = (tp - stp) / (0.015 * md.where(md > 0))
    return out


def build_ind() -> pd.DataFrame:
    if not os.path.exists(RAW):
        sys.exit(f'missing input: {RAW}')
    df = pd.read_csv(RAW, parse_dates=['Date']).sort_values(['Ticker', 'Date']).drop_duplicates(['Ticker', 'Date'], keep='last')
    ok = (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0) & (df['Open'] > 0) & (df['Adj Close'] > 0) & (df['High'] >= df['Low'])
    df = df[ok].copy()
    f = df['Adj Close'] / df['Close']
    df['P'], df['H'], df['L'] = df['Adj Close'], df['High'] * f, df['Low'] * f
    df['V'] = df['Volume'].where(df['Volume'] > 0)
    ind = pd.concat([one_stock(g) for _, g in df.groupby('Ticker', sort=False)], ignore_index=True)
    ind[FEATS] = ind[FEATS].replace([np.inf, -np.inf], np.nan).astype('float32')
    ind.to_pickle(IND)
    print('indicators:', ind.shape, '| missing share per feature:', ind[FEATS].isna().mean().round(4).to_dict())
    return ind


def frame() -> pd.DataFrame:
    """Panel rows with the 15 indicators of day t, the price of day t and the target price of day t+1."""
    if not os.path.exists(IND):
        build_ind()
    panel = C.load_panel()[['Ticker', 'Date', 'P', 'fut_ret', 'y']]
    d = panel.merge(pd.read_pickle(IND), on=['Ticker', 'Date'], how='left', validate='one_to_one')
    d['target'] = d['P'].astype('float64') * (1 + d['fut_ret'].astype('float64'))
    return d


def fit_one(tr: pd.DataFrame, te: pd.DataFrame, target='price'):
    """Returns the predicted return of each test row (predicted price / price of day t - 1)."""
    ytr = tr['target'].to_numpy() if target == 'price' else tr['fut_ret'].to_numpy(dtype='float64')
    m = xgb.XGBRegressor(**XGB_PARAMS).fit(tr[FEATS].to_numpy(), ytr)            # XGBoost handles missing values itself
    pred = m.predict(te[FEATS].to_numpy()).astype('float64')
    return pred / te['P'].to_numpy(dtype='float64') - 1 if target == 'price' else pred


def fit_a(target='price'):
    d = frame()
    rows = []
    for tk, g in d.groupby('Ticker', sort=True):
        g = g.sort_values('Date')
        if len(g) < N_TEST + MIN_TRAIN:
            continue
        tr, te = g.iloc[:-N_TEST], g.iloc[-N_TEST:]
        s = fit_one(tr, te, target)
        o = te[['Ticker', 'Date', 'y', 'P']].copy()
        o['score'] = s
        o['above_train_max'] = (te['P'].to_numpy() > tr['target'].max()).astype(float)
        rows.append(o)
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(CACHE, exist_ok=True)
    out.to_pickle(os.path.join(CACHE, f'A_{target}.pkl'))
    print('protocol A', target, '| tickers', out['Ticker'].nunique(), '| rows', len(out), '|', out['Date'].min().date(), '->', out['Date'].max().date())


def fit_b(years, target='price'):
    d = frame()
    panel_dates = d['Date']
    os.makedirs(CACHE, exist_ok=True)
    for yr, trm, tem in C.folds(pd.DataFrame({'Date': panel_dates})):
        if yr not in years:
            continue
        path = os.path.join(CACHE, f'B_{target}_{yr}.npy')
        if os.path.exists(path):
            print(yr, 'cached'); continue
        s = np.full(int(tem.sum()), np.nan)
        te_all = d[tem]
        pos = pd.Series(np.arange(len(te_all)), index=te_all.index)
        tr_all = d[trm]
        for tk, te in te_all.groupby('Ticker', sort=False):
            tr = tr_all[tr_all['Ticker'] == tk]
            if len(tr) < MIN_TRAIN:
                continue
            s[pos[te.index].to_numpy()] = fit_one(tr, te, target)
        np.save(path, s)
        print(yr, 'rows', len(s), 'covered', float(np.mean(~np.isnan(s))).__round__(4), flush=True)


def bal_acc(pred, y, w=None):
    """pred, y: arrays (0/1, NaN = not predicted). w: optional row weights. Balanced accuracy and plain accuracy."""
    ok = ~np.isnan(pred)
    w = np.ones(len(y)) if w is None else w
    w = w * ok
    p = np.where(ok, pred, 0)
    tpr = (w * (p == 1) * (y == 1)).sum() / max((w * (y == 1)).sum(), 1e-12)
    tnr = (w * (p == 0) * (y == 0)).sum() / max((w * (y == 0)).sum(), 1e-12)
    acc = (w * (p == y)).sum() / max(w.sum(), 1e-12)
    return 0.5 * (tpr + tnr), acc


def day_weights(dates, n_boot=5000, block=20, seed=7):
    days = np.sort(pd.unique(dates))
    idx = np.searchsorted(days, dates)
    rng = np.random.default_rng(seed)
    blk = np.arange(len(days)) // block
    nb = int(blk.max()) + 1
    draws = rng.integers(0, nb, size=(n_boot, nb))
    cnt = np.zeros((n_boot, nb), dtype=np.float32)
    np.add.at(cnt, (np.repeat(np.arange(n_boot), nb), draws.ravel()), 1)
    return idx, cnt[:, blk]                                                  # row -> day index, (n_boot, n_days) weights


def ticker_mean_ba(df, col, idx, W):
    """Mean over tickers of per-ticker balanced accuracy: point and bootstrap draws (days resampled in blocks)."""
    D = W.shape[1]
    pts, draws = [], []
    for tk, g in df.groupby('Ticker', sort=True):
        k = g.index.to_numpy()
        pred, y, di = g[col].to_numpy(dtype='float64'), g['y'].to_numpy(dtype='float64'), idx[k]
        ok = ~np.isnan(pred)
        if ok.sum() < 50 or y[ok].min() == y[ok].max():
            continue
        f = lambda m: np.bincount(di[ok], weights=m[ok].astype('float64'), minlength=D).astype(np.float32)
        tp, pos = f((pred == 1) & (y == 1)), f(y == 1)
        tn, neg = f((pred == 0) & (y == 0)), f(y == 0)
        pts.append(0.5 * (tp.sum() / pos.sum() + tn.sum() / neg.sum()))
        draws.append(0.5 * ((W @ tp) / np.maximum(W @ pos, 1e-9) + (W @ tn) / np.maximum(W @ neg, 1e-9)))
    return float(np.mean(pts)), np.mean(np.vstack(draws), axis=0), len(pts)


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate():
    panel = C.load_panel()
    tf = C.test_frame(panel)
    p_lgb, thr = C.load_pred('lgb_T27')
    tf['lgb_dir'] = np.where(np.isnan(p_lgb), np.nan, (p_lgb > thr).astype(float))
    tf['lgb_p'] = p_lgb
    summ, per_ticker = [], None
    # ---------------- protocol A: last 300 rows per ticker, per-ticker balanced accuracy averaged (the paper's metric)
    for target in ('price', 'ret'):
        path = os.path.join(CACHE, f'A_{target}.pkl')
        if not os.path.exists(path):
            continue
        a = pd.read_pickle(path).merge(tf[['Ticker', 'Date', 'lgb_dir', 'lgb_p']], on=['Ticker', 'Date'], how='left').reset_index(drop=True)
        a['dir'] = (a['score'] > 0).astype(float)
        a['up'] = 1.0
        idx, W = day_weights(a['Date'].to_numpy())
        pt, dr, n = ticker_mean_ba(a, 'dir', idx, W)
        pl, dl, _ = ticker_mean_ba(a, 'lgb_dir', idx, W)
        y = a['y'].to_numpy(dtype='float64')
        row = {'protocol': 'A: last 300 rows per ticker', 'target': target, 'tickers': n, 'rows': len(a),
               'bal_acc_ticker_mean': pt, 'ba_lo': ci(dr)[0], 'ba_hi': ci(dr)[1],
               'accuracy_pooled': float((a['dir'] == a['y']).mean()), 'pred_up_share': float(a['dir'].mean()), 'actual_up_share': float(y.mean()),
               'auroc_pooled': C._auc(y, a['score'].to_numpy()), 'rows_price_above_train_max': float(a['above_train_max'].mean()),
               'lgb_T27_bal_acc_ticker_mean': pl, 'lgb_lo': ci(dl)[0], 'lgb_hi': ci(dl)[1],
               'lgb_T27_accuracy_pooled': float((a['lgb_dir'] == a['y']).mean()), 'lgb_T27_auroc_pooled': C._auc(y, a['lgb_p'].to_numpy()),
               'diff_vs_lgb': pt - pl, 'diff_lo': ci(dr - dl)[0], 'diff_hi': ci(dr - dl)[1], 'paper_baseline': 0.5652}
        summ.append(row)
        if target == 'price':
            per_ticker = a.groupby('Ticker').apply(lambda g: pd.Series({
                'bal_acc': bal_acc(g['dir'].to_numpy(), g['y'].to_numpy(dtype='float64'))[0], 'pred_up_share': g['dir'].mean(),
                'above_train_max': g['above_train_max'].mean(), 'lgb_T27_bal_acc': bal_acc(g['lgb_dir'].to_numpy(dtype='float64'), g['y'].to_numpy(dtype='float64'))[0]}))
            per_ticker.to_csv(os.path.join(C.RES, 'relwork_mukherjee_per_ticker.csv'))
    # ---------------- protocol B: our yearly walk-forward on all 2018+ panel rows
    for target in ('price', 'ret'):
        parts = [os.path.join(CACHE, f'B_{target}_{yr}.npy') for yr in C.TEST_YEARS]
        if not all(os.path.exists(x) for x in parts):
            continue
        s = np.concatenate([np.load(x) for x in parts])
        if len(s) != len(tf):
            sys.exit('walk-forward cache does not match the test frame')
        b = tf[['Ticker', 'Date', 'y', 'lgb_dir', 'lgb_p']].copy()
        b['score'] = s
        b['dir'] = np.where(np.isnan(s), np.nan, (s > 0).astype(float))
        keep = b['dir'].notna() & b['lgb_dir'].notna()
        b = b[keep].reset_index(drop=True)
        y = b['y'].to_numpy(dtype='float64')
        idx, W = day_weights(b['Date'].to_numpy(), n_boot=2000)
        D = W.shape[1]

        def pooled(col):
            pr = b[col].to_numpy(dtype='float64')
            f = lambda m: np.bincount(idx, weights=m.astype('float64'), minlength=D).astype(np.float32)
            tp, pos, tn, neg = f((pr == 1) & (y == 1)), f(y == 1), f((pr == 0) & (y == 0)), f(y == 0)
            ba = 0.5 * (tp.sum() / pos.sum() + tn.sum() / neg.sum())
            bad = 0.5 * ((W @ tp) / (W @ pos) + (W @ tn) / (W @ neg))
            acc = (tp.sum() + tn.sum()) / len(y)
            accd = (W @ (tp + tn)) / (W @ (pos + neg))
            return ba, bad, acc, accd

        ba, bad, acc, accd = pooled('dir')
        bl, bld, al, ald = pooled('lgb_dir')
        boot = C.Boot(b, n_boot=2000)
        au, aud = boot.pooled(y, b['score'].to_numpy())
        aul, auld = boot.pooled(y, b['lgb_p'].to_numpy())
        pdm = float(C.day_auc(b, b['score'].to_numpy()).mean())
        pdl = float(C.day_auc(b, b['lgb_p'].to_numpy()).mean())
        summ.append({'protocol': 'B: yearly walk-forward 2018-2026', 'target': target, 'tickers': int(b['Ticker'].nunique()), 'rows': len(b),
                     'bal_acc_pooled': ba, 'ba_lo': ci(bad)[0], 'ba_hi': ci(bad)[1], 'accuracy_pooled': acc, 'acc_lo': ci(accd)[0], 'acc_hi': ci(accd)[1],
                     'pred_up_share': float(b['dir'].mean()), 'actual_up_share': float(y.mean()),
                     'auroc_pooled': au, 'auroc_lo': ci(aud)[0], 'auroc_hi': ci(aud)[1], 'perday_auc': pdm,
                     'lgb_T27_bal_acc_pooled': bl, 'lgb_lo': ci(bld)[0], 'lgb_hi': ci(bld)[1], 'lgb_T27_accuracy_pooled': al,
                     'lgb_T27_auroc_pooled': aul, 'lgb_T27_perday_auc': pdl,
                     'diff_vs_lgb': ba - bl, 'diff_lo': ci(bad - bld)[0], 'diff_hi': ci(bad - bld)[1],
                     'auroc_diff_vs_lgb': au - aul, 'auroc_diff_lo': ci(aud - auld)[0], 'auroc_diff_hi': ci(aud - auld)[1]})
    out = pd.DataFrame(summ)
    out.to_csv(os.path.join(C.RES, 'relwork_mukherjee_summary.csv'), index=False)
    json.dump({'features': FEATS, 'xgb_params': XGB_PARAMS, 'n_test': N_TEST, 'min_train': MIN_TRAIN,
               'assumptions': ['one model per ticker', 'XGBoost defaults', 'train on all earlier rows of the ticker',
                               'standard Schaff Trend Cycle with Table 2 windows', 'adjusted prices, panel rows']},
              open(os.path.join(C.RES, 'relwork_mukherjee_setup.json'), 'w'), indent=1)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
    print(out.T.to_string())
    if per_ticker is not None:
        print(per_ticker.describe().round(4).to_string())


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'ind':
        build_ind()
    elif cmd == 'fitA':
        fit_a(sys.argv[2] if len(sys.argv) > 2 else 'price')
    elif cmd == 'fitB':
        tgt = 'ret' if 'ret' in sys.argv[2:] else 'price'
        fit_b([int(a) for a in sys.argv[2:] if a.isdigit()], tgt)
    elif cmd == 'eval':
        evaluate()
    else:
        sys.exit(__doc__)
