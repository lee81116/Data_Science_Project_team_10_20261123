"""Spec 1 - timing controls. Global mode, common training period (2017+). Usage: 02_spec1_fit.py [lgb|lr]"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CONFIGS = {
    'C1_VIXCLOSE': C.T27 + ['vixc_chg', 'vixc_z60'],      # same indicator, known at close of t
    'C3_VIXPRE_LAG': C.T27 + ['gap_lag', 'gap_lag_z60'],  # placebo A: gap shifted by one day
    'C4_VIXPRE_SHUF': C.T27 + ['gap_shuf', 'gap_shuf_z60'],  # placebo B: 20-day block shuffle of dates
    'C5_ONRET': C.T27 + ['on_ret'],                       # upper bound: stock's own overnight return (known at open)
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'lgb'
    panel = C.load_panel()
    tf = C.test_frame(panel)
    rows, by_year = [], []
    names = {'C0_T27': f'{which}_T27_common', 'C2_VIXPRE': f'{which}_T27+VIXPRE_common'}
    for name, feats in CONFIGS.items():
        print(f'[{which}] {name}', flush=True)
        p, thr = C.run_config(panel, feats, which, '2017-01-01', None, verbose=False)
        C.save_pred(f'{which}_{name}', p, thr)
        names[name] = f'{which}_{name}'
    for name in ['C0_T27', 'C1_VIXCLOSE', 'C3_VIXPRE_LAG', 'C4_VIXPRE_SHUF', 'C2_VIXPRE', 'C5_ONRET']:
        p, thr = C.load_pred(names[name])
        ev = C.evaluate(tf, p, thr)
        b = ev.pop('by_year'); b.insert(0, 'config', name); b.insert(0, 'model', which); by_year.append(b)
        rows.append({'model': which, 'config': name, **ev})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(C.RES, f'spec1_summary_{which}.csv'), index=False)
    pd.concat(by_year).to_csv(os.path.join(C.RES, f'spec1_by_year_{which}.csv'), index=False)
    cols = ['config', 'auroc_pooled', 'auroc_yavg', 'perday_pooled', 'auroc_on_pooled', 'auroc_id_pooled', 'accuracy_yavg']
    print(summ[cols].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
