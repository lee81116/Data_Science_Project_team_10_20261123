"""Technical-only LSTM of Wang et al. (2026, Sci Rep 16:17494, FusionLSTM-CNF) re-run on our stocks.

What is taken from the paper: 23 price-volume and technical inputs, 60-day sequences, two stacked LSTM layers with
hidden size 128, dropout 0.3, layer normalisation, a fully connected head, AdamW with learning rate 0.001, cosine
annealing with warm restarts, batch size 64, at most 100 epochs, early stopping after 15 epochs without improvement of
the validation loss, gradient clipping at 1.0, inputs standardised with training-set statistics, and the split dates
(train 2020-01 -> 2023-04, validation 2023-05 -> 2024-02, test 2024-03 -> 2024-12).
The paper reports, on 30 S&P 500 stocks: LSTM baseline accuracy 0.523 / AUC 0.567 (Table 3, inputs not described),
technical-only branch accuracy 0.534 / AUC 0.578 (Table 6).

What the paper does not state, and what is assumed here:
  * indicator windows other than the moving averages (standard defaults: RSI 14, MACD 12/26/9, Bollinger 20 and 2 sd,
    ATR 14, stochastic 14/3, ADX 14, CCI 20, Williams %R 14, momentum 10);
  * one model for all stocks, inputs standardised per stock with that stock's training-period mean and sd;
  * head = LayerNorm -> dropout -> Linear(128, 64) -> ReLU -> dropout -> Linear(64, 1), binary cross-entropy,
    no label smoothing, AdamW weight decay at the library default, warm restarts every 10 epochs;
  * label (ours): y = 1 if AdjClose(t+1) > AdjClose(t); pooled accuracy at 0.5 and pooled AUROC.
The news branch, the cross-asset branch and the confidence-aware fusion are NOT re-run: we have no news data.

Usage (from final_experiments/scripts, needs torch):
  python3 20_relwork_wang.py prep                 # sequences -> preds/wang/data.npz
  python3 20_relwork_wang.py fit 42 [max_epochs]  # one training run, predictions -> preds/wang/lstm_seed42.npz
  python3 20_relwork_wang.py eval                 # -> results/relwork_wang_{summary,runs}.csv
Environment overrides: WANG_RAW, WANG_INPUTS (panel export with LightGBM predictions), WANG_OUT.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.environ.get('WANG_RAW', os.path.join(ROOT, '..', 'dataset', 'archive', 'NASDAQ100_Historical_Data.csv'))
INPUTS = os.environ.get('WANG_INPUTS', os.path.join(ROOT, 'data', 'wang_inputs.pkl'))
OUT = os.environ.get('WANG_OUT', ROOT)
CACHE, RES = os.path.join(OUT, 'preds', 'wang'), os.path.join(OUT, 'results')
SEQ = 60
TRAIN = ('2020-01-01', '2023-04-30')
VAL = ('2023-05-01', '2024-02-29')
TEST = ('2024-03-01', '2024-12-31')
FEATS = ['open', 'high', 'low', 'close', 'volume', 'sma5', 'sma10', 'sma20', 'sma50', 'rsi', 'macd', 'macd_signal',
         'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'obv', 'stoch_k', 'stoch_d', 'adx', 'cci', 'willr', 'momentum']
HP = dict(hidden=128, layers=2, dropout=0.3, lr=1e-3, batch=64, max_epochs=100, patience=15, clip=1.0, t0=10)


def wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def one_stock(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values('Date')
    o, h, l, c, v = g['O'], g['H'], g['L'], g['P'], g['V']
    f = pd.DataFrame({'Ticker': g['Ticker'], 'Date': g['Date'], 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
    for n in (5, 10, 20, 50):
        f[f'sma{n}'] = c.rolling(n, min_periods=n).mean()
    d = c.diff()
    f['rsi'] = 100 - 100 / (1 + wilder(d.clip(lower=0), 14) / wilder(-d.clip(upper=0), 14))
    macd = c.ewm(span=12, adjust=False, min_periods=12).mean() - c.ewm(span=26, adjust=False, min_periods=26).mean()
    f['macd'] = macd
    f['macd_signal'] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    mid, sd = c.rolling(20, min_periods=20).mean(), c.rolling(20, min_periods=20).std()
    f['bb_upper'], f['bb_middle'], f['bb_lower'] = mid + 2 * sd, mid, mid - 2 * sd
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    f['atr'] = wilder(tr, 14)
    f['obv'] = (np.sign(d).fillna(0) * v.fillna(0)).cumsum()
    hh, ll = h.rolling(14, min_periods=14).max(), l.rolling(14, min_periods=14).min()
    rng = (hh - ll).where((hh - ll) > 0)
    f['stoch_k'] = 100 * (c - ll) / rng
    f['stoch_d'] = f['stoch_k'].rolling(3, min_periods=3).mean()
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    atr = wilder(tr, 14)
    pdi, ndi = 100 * wilder(pdm, 14) / atr, 100 * wilder(ndm, 14) / atr
    f['adx'] = wilder(100 * (pdi - ndi).abs() / (pdi + ndi).where((pdi + ndi) > 0), 14)
    tp = (h + l + c) / 3
    stp = tp.rolling(20, min_periods=20).mean()
    md = tp.rolling(20, min_periods=20).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True)
    f['cci'] = (tp - stp) / (0.015 * md.where(md > 0))
    f['willr'] = -100 * (hh - c) / rng
    f['momentum'] = c - c.shift(10)
    return f


def prep():
    for p in (RAW, INPUTS):
        if not os.path.exists(p):
            sys.exit(f'missing input: {p}')
    df = pd.read_csv(RAW, parse_dates=['Date']).sort_values(['Ticker', 'Date']).drop_duplicates(['Ticker', 'Date'], keep='last')
    df = df[df['Date'] >= '2018-06-01']
    ok = (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0) & (df['Open'] > 0) & (df['Adj Close'] > 0) & (df['High'] >= df['Low'])
    df = df[ok].copy()
    k = df['Adj Close'] / df['Close']
    df['P'], df['H'], df['L'], df['O'] = df['Adj Close'], df['High'] * k, df['Low'] * k, df['Open'] * k
    df['V'] = df['Volume'].astype('float64')
    feat = pd.concat([one_stock(g) for _, g in df.groupby('Ticker', sort=False)], ignore_index=True)
    feat[FEATS] = feat[FEATS].replace([np.inf, -np.inf], np.nan)
    inp = pd.read_pickle(INPUTS)
    lab = inp[['Ticker', 'Date', 'y', 'lgb_p', 'lgb_thr']]
    X = {s: [] for s in ('train', 'val', 'test')}
    meta = {s: [] for s in ('train', 'val', 'test')}
    last = []                                             # last-day features of every sample, for the matched-input tree model
    for tk, g in feat.groupby('Ticker', sort=True):
        g = g.sort_values('Date').reset_index(drop=True)
        F = g[FEATS].to_numpy(dtype='float64')
        dates = g['Date'].to_numpy()
        tr = (dates >= np.datetime64(TRAIN[0])) & (dates <= np.datetime64(TRAIN[1]))
        if tr.sum() < 200:
            continue
        mu, sd = np.nanmean(F[tr], axis=0), np.nanstd(F[tr], axis=0)
        Z = ((F - mu) / np.where(sd > 0, sd, 1.0)).astype(np.float32)
        full = np.isfinite(Z).all(axis=1)
        okwin = np.convolve(full.astype(int), np.ones(SEQ, dtype=int), mode='full')[:len(full)] == SEQ   # 60 complete rows ending at t
        m = g[['Ticker', 'Date']].merge(lab, on=['Ticker', 'Date'], how='left')
        for s, (a, b) in (('train', TRAIN), ('val', VAL), ('test', TEST)):
            sel = np.where(okwin & (dates >= np.datetime64(a)) & (dates <= np.datetime64(b)) & m['y'].notna().to_numpy())[0]
            if s in ('train', 'val') and len(sel):
                sel = sel[dates[sel] < dates[sel].max()]   # the label of the last day uses the first close of the next segment
            if not len(sel):
                continue
            X[s].append(np.stack([Z[t - SEQ + 1:t + 1] for t in sel]))
            mm = m.iloc[sel].copy()
            mm['split'] = s
            meta[s].append(mm)
            last.append(pd.DataFrame(F[sel], columns=FEATS).assign(Ticker=tk, Date=dates[sel], split=s))
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for s in X:
        out[f'x_{s}'] = np.concatenate(X[s]).astype(np.float32)
        mt = pd.concat(meta[s], ignore_index=True)
        mt.to_pickle(os.path.join(CACHE, f'meta_{s}.pkl'))
        out[f'y_{s}'] = mt['y'].to_numpy(dtype=np.float32)
        print(s, out[f'x_{s}'].shape, 'tickers', mt['Ticker'].nunique(), 'days', mt['Date'].nunique(), mt['Date'].min().date(), '->', mt['Date'].max().date(),
              'up-rate', round(float(mt['y'].mean()), 4))
    np.savez(os.path.join(CACHE, 'data.npz'), **out)
    pd.concat(last, ignore_index=True).to_pickle(os.path.join(CACHE, 'last_day_features.pkl'))


def fit(seed=42, max_epochs=None):
    import torch
    from torch import nn
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(int(os.environ.get('WANG_THREADS', '1')))
    d = np.load(os.path.join(CACHE, 'data.npz'))
    xtr, ytr = torch.from_numpy(d['x_train']), torch.from_numpy(d['y_train'])
    xva, yva = torch.from_numpy(d['x_val']), torch.from_numpy(d['y_val'])
    xte = torch.from_numpy(d['x_test'])

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(len(FEATS), HP['hidden'], num_layers=HP['layers'], dropout=HP['dropout'], batch_first=True)
            self.norm = nn.LayerNorm(HP['hidden'])
            self.head = nn.Sequential(nn.Dropout(HP['dropout']), nn.Linear(HP['hidden'], 64), nn.ReLU(), nn.Dropout(HP['dropout']), nn.Linear(64, 1))

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(self.norm(o[:, -1])).squeeze(-1)

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=HP['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=HP['t0'])
    lossf = nn.BCEWithLogitsLoss()

    def predict(x):
        net.eval()
        with torch.no_grad():
            return torch.cat([torch.sigmoid(net(x[i:i + 4096])) for i in range(0, len(x), 4096)]).numpy()

    def val_loss():
        net.eval()
        with torch.no_grad():
            tot = sum(lossf(net(xva[i:i + 4096]), yva[i:i + 4096]).item() * len(xva[i:i + 4096]) for i in range(0, len(xva), 4096))
        return float(tot / len(xva))

    best, best_state, bad, hist = np.inf, None, 0, []
    n, bs = len(xtr), HP['batch']
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for ep in range(max_epochs or HP['max_epochs']):
        net.train()
        perm = torch.randperm(n, generator=g)
        run = 0.0
        for b, i in enumerate(range(0, n, bs)):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(xtr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), HP['clip'])
            opt.step()
            sched.step(ep + i / n)
            run += loss.item() * len(idx)
        vl = val_loss()
        hist.append({'epoch': ep + 1, 'train_loss': run / n, 'val_loss': vl, 'minutes': (time.time() - t0) / 60})
        print(f"seed {seed} epoch {ep + 1} train {run / n:.5f} val {vl:.5f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= HP['patience']:
                break
    net.load_state_dict(best_state)
    np.savez(os.path.join(CACHE, f'lstm_seed{seed}.npz'), p_test=predict(xte), p_val=predict(xva), best_val=best,
             best_epoch=int(np.argmin([h['val_loss'] for h in hist])) + 1, epochs=len(hist))
    pd.DataFrame(hist).to_csv(os.path.join(CACHE, f'history_seed{seed}.csv'), index=False)
    print('done seed', seed, 'best epoch', int(np.argmin([h['val_loss'] for h in hist])) + 1, 'of', len(hist))


def evaluate():
    import glob

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    mt = pd.read_pickle(os.path.join(CACHE, 'meta_test.pkl')).reset_index(drop=True)
    y = mt['y'].to_numpy(dtype='float64')
    days = np.sort(mt['Date'].unique())
    di = np.searchsorted(days, mt['Date'].to_numpy())
    rng = np.random.default_rng(7)
    blk = np.arange(len(days)) // 20
    nb = int(blk.max()) + 1
    rows_of_block = [np.where(blk[di] == b)[0] for b in range(nb)]
    draws = [np.concatenate([rows_of_block[b] for b in rng.integers(0, nb, nb)]) for _ in range(2000)]

    def perday(s):
        d = pd.DataFrame({'d': di, 'y': y, 's': s})
        d['r'] = d.groupby('d')['s'].rank()
        g = d.groupby('d')
        n1, n = g['y'].sum(), g['y'].size()
        a = ((d['r'] * d['y']).groupby(d['d']).sum() - n1 * (n1 + 1) / 2) / (n1 * (n - n1))
        return float(a[(n1 > 0) & (n1 < n)].mean())

    def stats(s, thr=0.5):
        pred = (s > thr).astype(float)
        f = lambda idx: (roc_auc_score(y[idx], s[idx]), float((pred[idx] == y[idx]).mean()),
                         0.5 * (float(pred[idx][y[idx] == 1].mean()) + float(1 - pred[idx][y[idx] == 0].mean())))
        pt = f(np.arange(len(y)))
        bs = np.array([f(ix) for ix in draws])
        return pt, bs

    rows, boots = [], {}

    def add(name, s, thr=0.5, **kw):
        ok = ~np.isnan(s)
        if not ok.all():
            sys.exit(f'{name}: missing predictions on the test rows')
        pt, bs = stats(s, thr)
        boots[name] = bs
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        rows.append({'model': name, 'rows': len(y), 'auroc': pt[0], 'auroc_lo': lo[0], 'auroc_hi': hi[0], 'accuracy': pt[1], 'acc_lo': lo[1], 'acc_hi': hi[1],
                     'bal_acc': pt[2], 'ba_lo': lo[2], 'ba_hi': hi[2], 'perday_auc': perday(s), 'pred_up_share': float((s > thr).mean()), **kw})

    runs = []
    P = []
    for path in sorted(glob.glob(os.path.join(CACHE, 'lstm_seed*.npz'))):
        z = np.load(path)
        seed = int(os.path.basename(path)[9:-4])
        P.append(z['p_test'])
        pt, _ = stats(z['p_test'])
        runs.append({'seed': seed, 'best_epoch': int(z['best_epoch']), 'epochs': int(z['epochs']), 'best_val_loss': float(z['best_val']), 'test_auroc': pt[0], 'test_accuracy': pt[1],
                     'test_bal_acc': pt[2], 'pred_up_share': float((z['p_test'] > 0.5).mean())})
    if not P:
        sys.exit('no LSTM runs found')
    runs = pd.DataFrame(runs)
    runs.to_csv(os.path.join(RES, 'relwork_wang_runs.csv'), index=False)
    best = int(runs['best_val_loss'].idxmin())
    add('LSTM technical-only, run with the lowest validation loss', P[best], seed=int(runs.loc[best, 'seed']))
    add(f'LSTM technical-only, mean of {len(P)} runs (ensemble)', np.mean(P, axis=0))
    # matched-input tree model: last-day values of the same 23 features, same training window, our fixed LightGBM settings
    lf = pd.read_pickle(os.path.join(CACHE, 'last_day_features.pkl'))
    key = lambda d: d['Ticker'].astype(str) + '|' + pd.to_datetime(d['Date']).dt.strftime('%Y-%m-%d')
    lab = pd.concat([pd.read_pickle(os.path.join(CACHE, f'meta_{s}.pkl')) for s in ('train', 'val', 'test')], ignore_index=True)
    lf = lf.merge(lab[['Ticker', 'Date', 'y']], on=['Ticker', 'Date'], how='left')
    rel = lf.copy()                                        # price-level inputs made relative to the close, as a tree model needs
    for c in ('open', 'high', 'low', 'sma5', 'sma10', 'sma20', 'sma50', 'bb_upper', 'bb_middle', 'bb_lower', 'macd', 'macd_signal', 'atr', 'momentum'):
        rel[c] = lf[c] / lf['close']
    params = dict(objective='binary', num_leaves=7, n_estimators=200, min_child_samples=1000, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=10.0, random_state=42, n_jobs=2, verbose=-1)
    for nm, dd, cols in (('LightGBM on the same 23 inputs (last day, raw levels)', lf, FEATS),
                         ('LightGBM on the same 23 inputs (last day, relative to the close)', rel, [c for c in FEATS if c not in ('close', 'volume', 'obv')])):
        tr = dd['split'].isin(['train', 'val'])
        te = dd[dd['split'] == 'test'].set_index(key(dd[dd['split'] == 'test'])).reindex(key(mt))
        m = lgb.LGBMClassifier(**params).fit(dd.loc[tr, cols].to_numpy(), dd.loc[tr, 'y'].to_numpy())
        add(nm, m.predict_proba(te[cols].to_numpy())[:, 1], thr=float(dd.loc[tr, 'y'].mean()))
    add('LightGBM T27, our walk-forward (trained 2002-2023)', mt['lgb_p'].to_numpy(dtype='float64'), thr=mt['lgb_thr'].to_numpy(dtype='float64'))
    out = pd.DataFrame(rows)
    ref = 'LightGBM T27, our walk-forward (trained 2002-2023)'
    for i, r in out.iterrows():
        dlt = boots[r['model']][:, 0] - boots[ref][:, 0]
        out.loc[i, 'auroc_minus_lgb_T27'] = r['auroc'] - out.loc[out['model'] == ref, 'auroc'].iloc[0]
        out.loc[i, 'diff_lo'], out.loc[i, 'diff_hi'] = np.percentile(dlt, [2.5, 97.5])
    out['actual_up_share'] = float(y.mean())
    out['paper_lstm_baseline_auc'], out['paper_technical_only_auc'] = 0.567, 0.578
    out.to_csv(os.path.join(RES, 'relwork_wang_summary.csv'), index=False)
    json.dump({'features': FEATS, 'hp': HP, 'seq': SEQ, 'train': TRAIN, 'val': VAL, 'test': TEST, 'test_days': int(len(days)), 'test_tickers': int(mt['Ticker'].nunique())},
              open(os.path.join(RES, 'relwork_wang_setup.json'), 'w'), indent=1)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(runs.round(4).to_string(index=False))
    print(out.drop(columns=['paper_lstm_baseline_auc', 'paper_technical_only_auc']).round(4).T.to_string())


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'prep':
        prep()
    elif cmd == 'fit':
        fit(int(sys.argv[2]) if len(sys.argv) > 2 else 42, int(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif cmd == 'eval':
        evaluate()
    else:
        sys.exit(__doc__)
