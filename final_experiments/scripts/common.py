"""Shared walk-forward, model, metric and bootstrap code (protocol follows EXPERIMENT_SETUP_REPORT.md)."""
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, PRED, RES, FIG = (os.path.join(ROOT, d) for d in ('data', 'preds', 'results', 'figures'))
TEST_YEARS = list(range(2018, 2027))
KRAUSS = [f'R_{m}' for m in list(range(1, 21)) + list(range(40, 241, 20))]
T27 = ['return_1d', 'return_5d', 'return_10d', 'return_20d',
       'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50', 'sma_ratio_200',
       'ema_ratio_12', 'ema_ratio_26', 'macd_hist', 'rsi_14', 'bb_position',
       'volatility_10d', 'volatility_20d', 'volatility_60d',
       'volume_sma_ratio_10', 'volume_sma_ratio_20',
       'daily_range', 'avg_range_10d', 'high_low_ratio',
       'momentum_10d', 'momentum_20d', 'roc_5', 'roc_10', 'atr_14']
VIXPRE = ['gap_next', 'gap_z60']
LGB_PARAMS = dict(objective='binary', num_leaves=7, n_estimators=200, min_child_samples=1000,
                  learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=10.0, random_state=42, n_jobs=4, verbose=-1)


def load_panel() -> pd.DataFrame:
    path = os.path.join(DATA, 'panel.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError('run 00_build_panel.py first')
    return pd.read_pickle(path)


def test_frame(panel: pd.DataFrame) -> pd.DataFrame:
    cols = ['Ticker', 'Date', 'y', 'y_on', 'y_id', 'fut_ret', 'on_ret', 'id_ret', 'gap_z60', 'gap_next', 'rho', 'mkt_ret']
    return panel.loc[panel['Date'] >= '2018-01-01', cols].reset_index(drop=True)


def folds(panel: pd.DataFrame, train_start=None):
    d = panel['Date']
    for yr in TEST_YEARS:
        te = (d.dt.year == yr).to_numpy()
        tr = (d < pd.Timestamp(yr, 1, 1))
        if train_start is not None:
            tr &= d >= pd.Timestamp(train_start)
        tr &= d < d[tr].max()          # 1-day purge: drop the last training trading day
        yield yr, tr.to_numpy(), te


def _stats(x):
    return np.nanmedian(x, axis=0), np.nanpercentile(x, 0.5, axis=0), np.nanpercentile(x, 99.5, axis=0)


def fit_predict(xtr, ytr, xte, model='lgb', seed=42, lgb_params=None):
    med, lo, hi = _stats(xtr)
    xtr, xte = np.clip(xtr, lo, hi), np.clip(xte, lo, hi)
    if model == 'lgb':
        params = dict(LGB_PARAMS if lgb_params is None else lgb_params)
        params['random_state'] = seed
        m = lgb.LGBMClassifier(**params).fit(xtr, ytr)          # NaN kept for LightGBM
        return m.predict_proba(xte)[:, 1]
    if model == 'lr':
        xtr = np.where(np.isnan(xtr), med, xtr)
        xte = np.where(np.isnan(xte), med, xte)
        sc = RobustScaler().fit(xtr)
        xtr, xte = np.clip(sc.transform(xtr), -10, 10), np.clip(sc.transform(xte), -10, 10)
        m = LogisticRegression(C=0.01, max_iter=1000).fit(xtr, ytr)
        return m.predict_proba(xte)[:, 1]
    raise ValueError(model)


def run_config(panel, feats, model='lgb', train_start=None, require=None, seed=42, label='y', verbose=True, lgb_params=None):
    """Walk-forward out-of-sample predictions on every 2018+ panel row. Returns (p, thr) aligned with test_frame()."""
    missing = [c for c in feats + [label] if c not in panel.columns]
    if missing:
        raise KeyError(f'missing columns: {missing}')
    x_all = panel[feats].to_numpy(dtype='float64')
    y_all = panel[label].to_numpy(dtype='float64')
    ok = ~np.isnan(y_all)
    if require is not None:
        ok_tr = ok & panel[require].notna().all(axis=1).to_numpy()
    else:
        ok_tr = ok
    is_test = (panel['Date'] >= '2018-01-01').to_numpy()
    p = np.full(len(panel), np.nan)
    thr = np.full(len(panel), np.nan)
    for yr, tr, te in folds(panel, train_start):
        tr = tr & ok_tr
        if tr.sum() < 1000:
            raise RuntimeError(f'{yr}: only {tr.sum()} training rows')
        p[te] = fit_predict(x_all[tr], y_all[tr], x_all[te], model, seed, lgb_params)
        thr[te] = y_all[tr].mean()
        if verbose:
            print(f'    {yr}: train {tr.sum():>7,}  test {te.sum():>6,}', flush=True)
    return p[is_test], thr[is_test]


def day_auc(tf: pd.DataFrame, score: np.ndarray, label='y') -> pd.Series:
    """Cross-sectional AUC within each day (NaN when a day has a single class)."""
    d = pd.DataFrame({'Date': tf['Date'].to_numpy(), 'y': tf[label].to_numpy(), 's': score})
    d = d.dropna()
    d['r'] = d.groupby('Date')['s'].rank(method='average')
    g = d.groupby('Date')
    n1 = g['y'].sum()
    n = g['y'].size()
    n0 = n - n1
    s1 = (d['r'] * d['y']).groupby(d['Date']).sum()
    auc = (s1 - n1 * (n1 + 1) / 2) / (n1 * n0)
    return auc.where((n1 > 0) & (n0 > 0))


def _auc(y, s):
    m = ~(np.isnan(y) | np.isnan(s))
    return roc_auc_score(y[m], s[m]) if m.sum() and 0 < y[m].mean() < 1 else np.nan


def evaluate(tf: pd.DataFrame, p: np.ndarray, thr: np.ndarray = None, label='y') -> dict:
    y = tf[label].to_numpy(dtype='float64')
    yrs = tf['Date'].dt.year.to_numpy()
    out = {'auroc_pooled': _auc(y, p)}
    da = day_auc(tf, p, label)
    out['perday_pooled'] = float(da.mean())
    rows = []
    for yr in TEST_YEARS:
        k = yrs == yr
        r = {'year': yr, 'n': int(k.sum()), 'auroc': _auc(y[k], p[k]),
             'perday': float(da[da.index.year == yr].mean()),
             'pr_auc': average_precision_score(y[k], p[k])}
        for lab in ('y_on', 'y_id'):
            if lab in tf:
                r[lab.replace('y_', 'auroc_')] = _auc(tf[lab].to_numpy(dtype='float64')[k], p[k])
        if thr is not None:
            pred = (p[k] > thr[k]).astype(float)
            yt = y[k]
            tp, tn = ((pred == 1) & (yt == 1)).sum(), ((pred == 0) & (yt == 0)).sum()
            fp, fn = ((pred == 1) & (yt == 0)).sum(), ((pred == 0) & (yt == 1)).sum()
            pu, ru = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            pdn, rdn = tn / max(tn + fn, 1), tn / max(tn + fp, 1)
            f1u = 2 * pu * ru / max(pu + ru, 1e-12)
            f1d = 2 * pdn * rdn / max(pdn + rdn, 1e-12)
            r.update(accuracy=(tp + tn) / k.sum(), macro_f1=(f1u + f1d) / 2, prec_up=pu, rec_up=ru,
                     prec_down=pdn, rec_down=rdn, pred_up_share=pred.mean())
        rows.append(r)
    by = pd.DataFrame(rows)
    out['by_year'] = by
    for c in by.columns:
        if c not in ('year', 'n'):
            out[c + '_yavg'] = float(by[c].mean())
    for lab in ('y_on', 'y_id'):
        if lab in tf:
            out[lab.replace('y_', 'auroc_') + '_pooled'] = _auc(tf[lab].to_numpy(dtype='float64'), p)
    return out


# ---------------------------------------------------------------- paired block bootstrap
class Boot:
    """Paired 20-trading-day block bootstrap over test days. Pooled AUROC via per-day score histograms."""

    def __init__(self, tf: pd.DataFrame, n_boot=10000, block=20, bins=1024, seed=7):
        days = np.sort(tf['Date'].unique())
        self.day_idx = np.searchsorted(days, tf['Date'].to_numpy())
        self.D = len(days)
        self.days = days
        rng = np.random.default_rng(seed)
        block_of_day = np.arange(self.D) // block
        nb = int(block_of_day.max()) + 1
        draws = rng.integers(0, nb, size=(n_boot, nb))
        counts = np.zeros((n_boot, nb), dtype=np.float32)
        np.add.at(counts, (np.repeat(np.arange(n_boot), nb), draws.ravel()), 1)
        w = counts[:, block_of_day]
        self.W = w
        self.bins = bins
        self.n_boot = n_boot

    def _hist(self, y, s):
        m = ~(np.isnan(y) | np.isnan(s))
        q = np.quantile(s[m], np.linspace(0, 1, self.bins + 1)[1:-1])
        b = np.searchsorted(np.unique(q), s[m], side='right')
        k = b.max() + 1
        key = self.day_idx[m] * k + b
        pos = np.bincount(key[y[m] == 1], minlength=self.D * k).reshape(self.D, k).astype(np.float32)
        neg = np.bincount(key[y[m] == 0], minlength=self.D * k).reshape(self.D, k).astype(np.float32)
        return pos, neg

    @staticmethod
    def _auc_from_hist(P, N):
        cn = np.cumsum(N, axis=1) - N
        return ((P * (cn + 0.5 * N)).sum(axis=1)) / (P.sum(axis=1) * N.sum(axis=1))

    def pooled(self, y, s):
        """Returns (point estimate, bootstrap draws) of pooled AUROC."""
        pos, neg = self._hist(np.asarray(y, dtype='float64'), np.asarray(s, dtype='float64'))
        point = self._auc_from_hist(pos.sum(0, keepdims=True), neg.sum(0, keepdims=True))[0]
        return float(point), self._auc_from_hist(self.W @ pos, self.W @ neg)

    def perday(self, da: pd.Series):
        """da: per-day AUC indexed by date. Returns (point, draws) of the mean over days."""
        v = pd.Series(da).reindex(pd.DatetimeIndex(self.days)).to_numpy(dtype='float64')
        ok = ~np.isnan(v)
        a = np.where(ok, v, 0).astype(np.float32)
        return float(np.nanmean(v)), (self.W @ a) / (self.W @ ok.astype(np.float32))


def delta_stats(a_draws, b_draws, a_point, b_point):
    """Stats for (b - a): point, 95% CI, two-sided bootstrap p."""
    d = b_draws - a_draws
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))
    return {'delta': b_point - a_point, 'lo': float(lo), 'hi': float(hi), 'p': float(min(p, 1.0))}


def holm(pvals):
    p = np.asarray(pvals, dtype='float64')
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def save_pred(name, p, thr=None):
    os.makedirs(PRED, exist_ok=True)
    np.save(os.path.join(PRED, f'{name}.npy'), np.vstack([p, thr if thr is not None else np.full(len(p), np.nan)]))


def load_pred(name):
    a = np.load(os.path.join(PRED, f'{name}.npy'))
    return a[0], a[1]
