"""Rebuild the analysis panel from raw OHLCV + Cboe VIX.

Follows EXPERIMENT_SETUP_REPORT.md (teammate) and features_transformation.py (27 indicators).
Output: data/panel.pkl, results/panel_row_accounting.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, '..', 'dataset', 'archive', 'NASDAQ100_Historical_Data.csv')
VIX = os.path.join(ROOT, '..', 'final_checks', 'external', 'vix_cboe_official.csv')
TRADING_DAYS = 252
KRAUSS_LAGS = list(range(1, 21)) + list(range(40, 241, 20))

T27 = ['return_1d', 'return_5d', 'return_10d', 'return_20d',
       'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50', 'sma_ratio_200',
       'ema_ratio_12', 'ema_ratio_26', 'macd_hist', 'rsi_14', 'bb_position',
       'volatility_10d', 'volatility_20d', 'volatility_60d',
       'volume_sma_ratio_10', 'volume_sma_ratio_20',
       'daily_range', 'avg_range_10d', 'high_low_ratio',
       'momentum_10d', 'momentum_20d', 'roc_5', 'roc_10', 'atr_14']


def one_stock(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values('Date').copy()
    p, h, l, v = g['P'], g['H'], g['L'], g['V']
    r1 = p.pct_change(1, fill_method=None)
    g['return_1d'] = r1
    for k in (5, 10, 20):
        g[f'return_{k}d'] = p.pct_change(k, fill_method=None)
    for w in (5, 10, 20, 50, 200):
        g[f'sma_ratio_{w}'] = p / p.rolling(w, min_periods=w).mean()
    e12 = p.ewm(span=12, adjust=False).mean()
    e26 = p.ewm(span=26, adjust=False).mean()
    g['ema_ratio_12'] = p / e12
    g['ema_ratio_26'] = p / e26
    macd = e12 - e26
    g['macd_hist'] = (macd - macd.ewm(span=9, adjust=False).mean()) / p
    d = p.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    g['rsi_14'] = 100 - 100 / (1 + ag / al)
    mid = p.rolling(20, min_periods=20).mean()
    sd = p.rolling(20, min_periods=20).std()
    g['bb_position'] = (p - mid) / (2 * sd)
    for w in (10, 20, 60):
        g[f'volatility_{w}d'] = r1.rolling(w, min_periods=w).std() * np.sqrt(TRADING_DAYS)
    for w in (10, 20):
        g[f'volume_sma_ratio_{w}'] = v / v.rolling(w, min_periods=w).mean()
    g['daily_range'] = (h - l) / p
    g['avg_range_10d'] = g['daily_range'].rolling(10, min_periods=10).mean()
    g['high_low_ratio'] = h / l
    g['momentum_10d'] = p / p.shift(10)
    g['momentum_20d'] = p / p.shift(20)
    g['roc_5'] = p.pct_change(5, fill_method=None)
    g['roc_10'] = p.pct_change(10, fill_method=None)
    pc = p.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    g['atr_14'] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / p
    # labels
    g['fut_ret'] = p.shift(-1) / p - 1
    g['on_ret'] = g['O'].shift(-1) / p - 1                      # adjusted overnight close(t)->open(t+1)
    g['on_ret_raw'] = g['Open'].shift(-1) / g['Close'] - 1      # raw (what the setup report describes)
    g['id_ret'] = g['Close'].shift(-1) / g['Open'].shift(-1) - 1  # raw intraday open(t+1)->close(t+1)
    for m in KRAUSS_LAGS:
        g[f'R_{m}'] = p / p.shift(m) - 1
    return g


def main() -> None:
    for path in (RAW, VIX):
        if not os.path.exists(path):
            sys.exit(f'missing input: {path}')
    acc = {}
    df = pd.read_csv(RAW, parse_dates=['Date'])
    acc['raw_rows'] = int(len(df))
    acc['raw_tickers'] = int(df['Ticker'].nunique())
    acc['raw_date_min'] = str(df['Date'].min().date())
    acc['raw_date_max'] = str(df['Date'].max().date())
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    acc['duplicate_ticker_date'] = int(df.duplicated(['Ticker', 'Date']).sum())
    df = df.drop_duplicates(['Ticker', 'Date'], keep='last')
    bad = ~((df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0) & (df['Open'] > 0)
            & (df['Adj Close'] > 0) & (df['High'] >= df['Low']))
    acc['invalid_ohlc_rows'] = int(bad.sum())
    df = df[~bad].copy()
    f = df['Adj Close'] / df['Close']
    df['P'] = df['Adj Close']
    df['H'] = df['High'] * f
    df['L'] = df['Low'] * f
    df['O'] = df['Open'] * f
    df['V'] = df['Volume'].where(df['Volume'] > 0)
    acc['zero_volume_rows'] = int((df['Volume'] <= 0).sum())

    out = pd.concat([one_stock(g) for _, g in df.groupby('Ticker', sort=False)], ignore_index=True)

    # market return (equal-weighted over all tickers with data that day) and leave-one-out rolling correlation
    wide = out.pivot(index='Date', columns='Ticker', values='return_1d')
    n = wide.notna().sum(axis=1)
    tot = wide.sum(axis=1, skipna=True)
    mkt = (tot / n.where(n > 0)).rename('mkt_ret')
    rho = {}
    for c in wide.columns:
        r = wide[c]
        others = n - r.notna().astype(int)
        ex = (tot - r.fillna(0)) / others.where(others > 0)
        rho[c] = r.rolling(252, min_periods=126).corr(ex)
    rho = pd.DataFrame(rho, index=wide.index).clip(-1, 1).stack().rename('rho').reset_index()
    rho.columns = ['Date', 'Ticker', 'rho']
    out = out.merge(rho, on=['Date', 'Ticker'], how='left').merge(mkt.reset_index(), on='Date', how='left')

    # VIX features (Cboe official daily)
    vix = pd.read_csv(VIX)
    vix['Date'] = pd.to_datetime(vix['DATE'], format='%m/%d/%Y')
    vix = vix.sort_values('Date').set_index('Date')
    if not vix.index.is_unique:
        sys.exit('VIX dates not unique')
    vf = pd.DataFrame(index=vix.index)
    gap_next = vix['OPEN'].shift(-1) / vix['CLOSE'] - 1            # known at 03:15 ET on t+1
    gap_z60 = gap_next / gap_next.shift(1).rolling(60).std()
    chg = vix['CLOSE'].pct_change(fill_method=None)                # known at close of t
    vf['gap_next'] = gap_next
    vf['gap_z60'] = gap_z60
    vf['vixc_chg'] = chg
    vf['vixc_z60'] = chg / chg.shift(1).rolling(60).std()
    vf['gap_lag'] = gap_next.shift(1)                              # = OPEN(t)/CLOSE(t-1)-1, known before close of t
    vf['gap_lag_z60'] = gap_z60.shift(1)
    vf['vix_close'] = vix['CLOSE']
    vf['gap_next_unmasked'] = gap_next
    rng = np.random.default_rng(42)
    valid = vf.index[(vf.index >= '2017-01-01') & gap_next.notna().to_numpy() & gap_z60.notna().to_numpy()]
    blocks = [valid[i:i + 20] for i in range(0, len(valid), 20)]
    src = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
    vf.loc[valid, 'gap_shuf'] = gap_next.loc[src].to_numpy()
    vf.loc[valid, 'gap_shuf_z60'] = gap_z60.loc[src].to_numpy()
    mask_cols = ['gap_next', 'gap_z60', 'vixc_chg', 'vixc_z60', 'gap_lag', 'gap_lag_z60', 'gap_shuf', 'gap_shuf_z60']
    vf.loc[vf.index < '2017-01-01', mask_cols] = np.nan
    out = out.merge(vf.reset_index(), on='Date', how='left')
    acc['stock_dates_without_vix'] = int(out.loc[out['vix_close'].isna(), 'Date'].nunique())

    # labels
    out['y'] = (out['fut_ret'] > 0).astype('float32').where(out['fut_ret'].notna())
    out['y_on'] = (out['on_ret'] > 0).astype('float32').where(out['on_ret'].notna())
    out['y_on_raw'] = (out['on_ret_raw'] > 0).astype('float32').where(out['on_ret_raw'].notna())
    out['y_id'] = (out['id_ret'] > 0).astype('float32').where(out['id_ret'].notna())

    # row filters, in the order of the setup report
    acc['rows_after_cleaning'] = int(len(out))
    m1 = out[T27].isna().any(axis=1) | out['V'].isna()
    acc['drop_missing_indicator_or_zero_volume'] = int(m1.sum())
    out = out[~m1]
    m2 = out['y'].isna()
    acc['drop_no_next_day_label'] = int(m2.sum())
    out = out[~m2]
    m3 = out['P'] < 5
    acc['drop_adj_close_below_5'] = int(m3.sum())
    out = out[~m3]
    m4 = out['Date'] < '2002-01-01'
    acc['drop_before_2002'] = int(m4.sum())
    out = out[~m4]
    acc['panel_rows'] = int(len(out))
    acc['panel_tickers'] = int(out['Ticker'].nunique())
    acc['panel_date_min'] = str(out['Date'].min().date())
    acc['panel_date_max'] = str(out['Date'].max().date())
    acc['test_rows_2018_2026'] = int((out['Date'] >= '2018-01-01').sum())
    acc['reported'] = {'raw_rows': 514075, 'drop_missing_or_zero_volume': 22887, 'drop_no_label': 100,
                       'drop_below_5': 48668, 'panel_rows': 430397, 'test_rows': 184061}

    keep = (['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'P', 'fut_ret', 'on_ret', 'on_ret_raw', 'id_ret',
             'y', 'y_on', 'y_on_raw', 'y_id', 'rho', 'mkt_ret'] + T27 + [f'R_{m}' for m in KRAUSS_LAGS]
            + mask_cols + ['vix_close', 'gap_next_unmasked'])
    out = out[keep].sort_values(['Date', 'Ticker']).reset_index(drop=True)
    num = out.select_dtypes('float64').columns
    out[num] = out[num].astype('float32')
    os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    out.to_pickle(os.path.join(ROOT, 'data', 'panel.pkl'))
    with open(os.path.join(ROOT, 'results', 'panel_row_accounting.json'), 'w') as fh:
        json.dump(acc, fh, indent=2)
    print(json.dumps(acc, indent=2))
    print('up rate:', round(float(out['y'].mean()), 4), '| cols:', out.shape[1])


if __name__ == '__main__':
    main()
