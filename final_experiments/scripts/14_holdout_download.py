"""Prospective holdout, step 1 - download prices dated after the Kaggle file ends (2026-02-18).
Source: Yahoo Finance chart API (daily bars, split-adjusted close + dividend-adjusted close), 2023-01-01 to today, for the 100 panel tickers.
        Cboe VIX daily history (official CSV).
Nothing here touches the existing panel. Output: data/holdout/yahoo_ohlcv.csv, data/holdout/vix_cboe_latest.csv, results/holdout_download_checks.json
Usage: python 14_holdout_download.py        (re-runnable; per-ticker responses are cached in data/holdout/raw/)
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = os.path.join(C.DATA, 'holdout')
RAWDIR = os.path.join(OUT, 'raw')
KAGGLE = os.path.join(C.ROOT, '..', 'dataset', 'archive', 'NASDAQ100_Historical_Data.csv')
VIX_OLD = os.path.join(C.ROOT, '..', 'final_checks', 'external', 'vix_cboe_official.csv')
START = '2023-01-01'
UA = {'User-Agent': 'Mozilla/5.0'}


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 + 3 * k)
        except Exception:
            time.sleep(2 + 3 * k)
    raise RuntimeError(f'failed after {tries} tries: {url[:80]}')


def one_ticker(sym, p1, p2):
    path = os.path.join(RAWDIR, f'{sym}.json')
    if not os.path.exists(path):
        body = get(f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d&events=div%2Csplits')
        if body is None:
            return None
        open(path, 'wb').write(body)
        time.sleep(0.25)
    res = json.load(open(path))['chart']['result']
    if not res or 'timestamp' not in res[0]:
        return None
    res = res[0]
    q = res['indicators']['quote'][0]
    d = pd.DataFrame({'Ticker': sym,
                      'Date': pd.to_datetime(res['timestamp'], unit='s', utc=True).tz_convert('America/New_York').tz_localize(None).normalize(),
                      'Open': q['open'], 'High': q['high'], 'Low': q['low'], 'Close': q['close'],
                      'Adj Close': res['indicators']['adjclose'][0]['adjclose'], 'Volume': q['volume']})
    ev = res.get('events', {})
    d.attrs['splits'] = [pd.Timestamp(int(k), unit='s').strftime('%Y-%m-%d') for k in ev.get('splits', {})]
    return d.dropna(subset=['Close', 'Adj Close'])


def main():
    os.makedirs(RAWDIR, exist_ok=True)
    kag = pd.read_csv(KAGGLE, parse_dates=['Date'])
    tickers = sorted(kag['Ticker'].unique())
    p1 = int(pd.Timestamp(START, tz='UTC').timestamp()); p2 = int(time.time())
    frames, missing, splits = [], [], {}
    for i, s in enumerate(tickers):
        d = one_ticker(s, p1, p2)
        if d is None or d.empty:
            missing.append(s)
            continue
        frames.append(d)
        late = [x for x in d.attrs['splits'] if x > '2026-02-18']
        if late:
            splits[s] = late
        if (i + 1) % 25 == 0:
            print(f'  {i + 1}/{len(tickers)} tickers', flush=True)
    y = pd.concat(frames, ignore_index=True)
    y.to_csv(os.path.join(OUT, 'yahoo_ohlcv.csv'), index=False)

    # consistency with the Kaggle file on the overlapping dates (returns are invariant to later dividend adjustments)
    k = kag[kag['Date'] >= START].sort_values(['Ticker', 'Date'])
    k['ret_k'] = k.groupby('Ticker')['Adj Close'].pct_change(fill_method=None)
    ys = y.sort_values(['Ticker', 'Date']).copy()
    ys['ret_y'] = ys.groupby('Ticker')['Adj Close'].pct_change(fill_method=None)
    m = k.merge(ys[['Ticker', 'Date', 'ret_y', 'Close', 'Volume']], on=['Ticker', 'Date'], how='inner', suffixes=('', '_y')).dropna(subset=['ret_k', 'ret_y'])
    diff = (m['ret_k'] - m['ret_y']).abs()
    last = ys.groupby('Ticker')['Date'].max()
    new = ys[ys['Date'] > kag['Date'].max()]
    vix_body = get('https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv')
    open(os.path.join(OUT, 'vix_cboe_latest.csv'), 'wb').write(vix_body)
    v_new = pd.read_csv(os.path.join(OUT, 'vix_cboe_latest.csv')); v_old = pd.read_csv(VIX_OLD)
    vm = v_old.merge(v_new, on='DATE', suffixes=('_old', '_new'))
    chk = {'downloaded_on': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()), 'tickers_requested': len(tickers), 'tickers_downloaded': int(y['Ticker'].nunique()),
           'tickers_missing': missing, 'kaggle_last_date': str(kag['Date'].max().date()), 'yahoo_last_date': str(y['Date'].max().date()),
           'tickers_not_reaching_last_date': sorted(last[last < y['Date'].max()].index.tolist()),
           'splits_after_2026_02_18': splits, 'new_rows_after_kaggle_end': int(len(new)), 'new_trading_days': int(new['Date'].nunique()),
           'overlap_rows_compared': int(len(m)), 'overlap_return_abs_diff_median': float(diff.median()), 'overlap_return_abs_diff_p999': float(diff.quantile(0.999)),
           'overlap_return_abs_diff_max': float(diff.max()), 'overlap_share_abs_diff_above_1e-4': float((diff > 1e-4).mean()),
           'overlap_return_corr': float(np.corrcoef(m['ret_k'], m['ret_y'])[0, 1]),
           'vix_old_last_date': str(v_old['DATE'].iloc[-1]), 'vix_new_last_date': str(v_new['DATE'].iloc[-1]), 'vix_overlap_days': int(len(vm)),
           'vix_overlap_max_abs_diff_open': float((vm['OPEN_old'] - vm['OPEN_new']).abs().max()), 'vix_overlap_max_abs_diff_close': float((vm['CLOSE_old'] - vm['CLOSE_new']).abs().max())}
    worst = m.assign(d=diff).sort_values('d', ascending=False).head(8)[['Ticker', 'Date', 'ret_k', 'ret_y', 'd']]
    chk['worst_overlap_rows'] = [{'Ticker': r.Ticker, 'Date': str(r.Date.date()), 'ret_kaggle': float(r.ret_k), 'ret_yahoo': float(r.ret_y)} for r in worst.itertuples()]
    json.dump(chk, open(os.path.join(C.RES, 'holdout_download_checks.json'), 'w'), indent=2)
    print(json.dumps(chk, indent=2))


if __name__ == '__main__':
    main()
