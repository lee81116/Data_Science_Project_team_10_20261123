"""Spec 4C - multi-seed reliability for the global LightGBM configs. Usage: 05_spec4_seeds.py <config>"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CONFIGS = {'T27': (C.T27, None), 'VIXPRE': (C.VIXPRE, C.VIXPRE), 'T27+VIXPRE': (C.T27 + C.VIXPRE, None)}


def main():
    name = sys.argv[1]
    feats, req = CONFIGS[name]
    panel = C.load_panel()
    tf = C.test_frame(panel)
    rows = []
    for seed in (42, 43, 44, 45, 46):
        p, thr = C.run_config(panel, feats, 'lgb', None, req, seed=seed, verbose=False)
        ev = C.evaluate(tf, p, thr)
        rows.append({'config': name, 'seed': seed, 'auroc_yavg': ev['auroc_yavg'], 'auroc_pooled': ev['auroc_pooled'],
                     'perday_yavg': ev['perday_yavg'], 'auroc_on_yavg': ev['auroc_on_yavg'], 'auroc_id_yavg': ev['auroc_id_yavg']})
        print(rows[-1], flush=True)
    out = os.path.join(C.RES, 'spec4_seeds.csv')
    df = pd.DataFrame(rows)
    if os.path.exists(out):
        old = pd.read_csv(out)
        df = pd.concat([old[old['config'] != name], df])
    df.to_csv(out, index=False)


if __name__ == '__main__':
    main()
