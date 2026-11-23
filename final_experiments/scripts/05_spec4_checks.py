"""Spec 4 - consistency and reliability checks that need the data."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    panel = C.load_panel()
    tf = C.test_frame(panel)
    out = {}

    # (b) class-wise precision / recall, global LightGBM, yearly average
    b = pd.read_csv(os.path.join(C.RES, 'baseline_by_year_lgb.csv'))
    cw = b[b['config'].isin(['T27', 'T27+VIXPRE'])].groupby('config')[
        ['accuracy', 'macro_f1', 'prec_up', 'rec_up', 'prec_down', 'rec_down', 'pred_up_share']].mean()
    cw.to_csv(os.path.join(C.RES, 'spec4_classwise.csv'))
    print('--- class-wise (yearly avg)'); print(cw.round(4).to_string())

    # (d) overlap years with the Progress protocol (99 stocks, 2010-2023 walk-forward)
    prog_path = os.path.join(C.ROOT, '..', 'progress_update', 'data', 'h1_per_year_auc.csv')
    if os.path.exists(prog_path):
        prog = pd.read_csv(prog_path)
        print('--- progress file columns:', list(prog.columns)); print(prog.head(3).to_string())
        prog.to_csv(os.path.join(C.RES, 'spec4_progress_h1_copy.csv'), index=False)
    mine = b[b['config'] == 'T27'][['year', 'auroc', 'perday', 'n']]
    mine.to_csv(os.path.join(C.RES, 'spec4_T27_by_year_new_protocol.csv'), index=False)

    # (e) when does the Cboe daily OPEN start to behave like a pre-market print?
    raw = pd.read_pickle(os.path.join(C.DATA, 'panel.pkl'))[['Date', 'on_ret', 'gap_next_unmasked']]
    day = raw.groupby('Date').agg(on=('on_ret', 'mean'), gap=('gap_next_unmasked', 'first')).dropna()
    day = day[(day.index >= '2013-01-01') & (day.index < '2019-01-01')]
    half = day.groupby([day.index.year, (day.index.month - 1) // 3 + 1]).apply(lambda d: pd.Series({'corr': d['on'].corr(d['gap']), 'days': len(d),
                                                                                              'gap_abs_mean': d['gap'].abs().mean()}))
    half.index.names = ['year', 'quarter']
    half.to_csv(os.path.join(C.RES, 'spec4_vix_open_by_quarter.csv'))
    print('--- corr(VIX open gap, mean stock overnight return) by quarter'); print(half.round(3).to_string())
    pre = day[day.index < '2016-04-15']; post = day[(day.index >= '2016-04-15') & (day.index < '2017-01-01')]; y17 = day[day.index >= '2017-01-01']
    out['vix_open_corr'] = {'2013-01..2016-04-14': float(pre['on'].corr(pre['gap'])), '2016-04-15..2016-12': float(post['on'].corr(post['gap'])),
                            '2017..2018': float(y17['on'].corr(y17['gap']))}
    print(out['vix_open_corr'])

    # (f) overnight label: adjusted vs raw prices (split days)
    te = panel[panel['Date'] >= '2018-01-01']
    diff = (te['y_on'] != te['y_on_raw']) & te['y_on'].notna() & te['y_on_raw'].notna()
    big = (te['on_ret_raw'] - te['on_ret']).abs() > 0.05
    out['overnight_label_rows_differ_adj_vs_raw'] = int(diff.sum())
    out['overnight_ret_gap_gt_5pct_rows'] = int(big.sum())
    p = C.load_pred('lgb_T27+VIXPRE')[0]
    out['auroc_on_adj_T27+VIXPRE'] = float(C._auc(tf['y_on'].to_numpy(dtype='float64'), p))
    out['auroc_on_raw_T27+VIXPRE'] = float(C._auc(panel.loc[panel['Date'] >= '2018-01-01', 'y_on_raw'].to_numpy(dtype='float64'), p))

    # (g) where does the structural model's cross-sectional skill sit? per-day AUC vs overnight / intraday labels
    boot = C.Boot(tf, n_boot=10000)
    rows = []
    for nm, f in (('M1_rho_x_gap', 'spec2_M1_rho_x_gap'), ('lgb_T27+VIXPRE_common', 'lgb_T27+VIXPRE_common'), ('lgb_C5_ONRET', 'lgb_C5_ONRET')):
        pp = C.load_pred(f)[0]
        for lab in ('y', 'y_on', 'y_id'):
            pt, dr = boot.perday(C.day_auc(tf, pp, lab))
            lo, hi = np.percentile(dr, [2.5, 97.5])
            rows.append({'model': nm, 'label': lab, 'perday_auc': pt, 'lo': lo, 'hi': hi})
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(C.RES, 'spec4_perday_by_segment.csv'), index=False)
    print('--- per-day AUC by label segment'); print(g.round(4).to_string(index=False))

    # (h) zero-parameter date-level baselines on the official label (pooled AUROC)
    rows = []
    base = boot.pooled(tf['y'].to_numpy(dtype='float64'), C.load_pred('lgb_T27')[0])
    for nm, s in (('minus prev-day market return', -tf['mkt_ret'].to_numpy(dtype='float64')),
                  ('minus pre-market VIX gap', -tf['gap_next'].to_numpy(dtype='float64'))):
        pt, dr = boot.pooled(tf['y'].to_numpy(dtype='float64'), s)
        lo, hi = np.percentile(dr, [2.5, 97.5])
        rows.append({'score': nm, 'auroc': pt, 'lo': lo, 'hi': hi, **{'d_vs_lgb_T27_' + k: v for k, v in C.delta_stats(base[1], dr, base[0], pt).items()}})
    h = pd.DataFrame(rows)
    h.to_csv(os.path.join(C.RES, 'spec4_zero_param_baselines.csv'), index=False)
    print('--- zero-parameter date-level scores'); print(h.round(4).to_string(index=False))

    with open(os.path.join(C.RES, 'spec4_checks.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
