"""Spec 1 - paired 20-day block bootstrap (10,000 draws). Usage: 02_spec1_boot.py [lgb|lr]"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FILES = {'C0': '{m}_T27_common', 'C1': '{m}_C1_VIXCLOSE', 'C2': '{m}_T27+VIXPRE_common',
         'C3': '{m}_C3_VIXPRE_LAG', 'C4': '{m}_C4_VIXPRE_SHUF', 'C5': '{m}_C5_ONRET'}
PAIRS = [('C0', 'C1'), ('C0', 'C3'), ('C0', 'C4'), ('C1', 'C2'), ('C2', 'C5')]   # Holm family (official label)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'lgb'
    tf = C.test_frame(C.load_panel())
    boot = C.Boot(tf, n_boot=10000)
    preds = {k: C.load_pred(v.format(m=which))[0] for k, v in FILES.items()}
    t0 = time.time()
    off = {k: boot.pooled(tf['y'].to_numpy(), p) for k, p in preds.items()}
    intr = {k: boot.pooled(tf['y_id'].to_numpy(), p) for k, p in preds.items()}
    pday = {k: boot.perday(C.day_auc(tf, p, 'y')) for k, p in preds.items()}
    print(f'bootstrap done in {time.time() - t0:.0f}s', flush=True)
    rows = []
    for a, b in PAIRS + [('C0', 'C2')]:
        r = {'model': which, 'label': 'official', 'metric': 'pooled_auroc', 'a': a, 'b': b,
             **C.delta_stats(off[a][1], off[b][1], off[a][0], off[b][0])}
        rows.append(r)
    fam = [r for r in rows if (r['a'], r['b']) in PAIRS]
    for r, ph in zip(fam, C.holm([r['p'] for r in fam])):
        r['p_holm'] = float(ph)
    for a, b in [('C0', 'C1'), ('C0', 'C2'), ('C0', 'C5'), ('C2', 'C5')]:
        rows.append({'model': which, 'label': 'intraday', 'metric': 'pooled_auroc', 'a': a, 'b': b,
                     **C.delta_stats(intr[a][1], intr[b][1], intr[a][0], intr[b][0])})
        rows.append({'model': which, 'label': 'official', 'metric': 'perday_auc', 'a': a, 'b': b,
                     **C.delta_stats(pday[a][1], pday[b][1], pday[a][0], pday[b][0])})
    pd.DataFrame(rows).to_csv(os.path.join(C.RES, f'spec1_deltas_{which}.csv'), index=False)
    lev = []
    for k in FILES:
        for lab, d in (('official', off), ('intraday', intr), ('perday_official', pday)):
            lo, hi = np.percentile(d[k][1], [2.5, 97.5])
            lev.append({'model': which, 'config': k, 'label': lab, 'value': d[k][0], 'lo': lo, 'hi': hi})
    pd.DataFrame(lev).to_csv(os.path.join(C.RES, f'spec1_levels_{which}.csv'), index=False)
    pd.set_option('display.width', 200)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(lev).pivot(index='config', columns='label', values=['value', 'lo', 'hi']).round(4).to_string())


if __name__ == '__main__':
    main()
