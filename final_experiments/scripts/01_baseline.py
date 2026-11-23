"""Reproduce the teammate's global-mode baseline numbers (EXPERIMENT_FULL_REPORT.md, experiments 1, 4, 5)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CONFIGS = {  # name: (features, train_start, require)
    'T27': (C.T27, None, None),
    'VIXPRE': (C.VIXPRE, None, C.VIXPRE),
    'T27+VIXPRE': (C.T27 + C.VIXPRE, None, None),
    'T27_common': (C.T27, '2017-01-01', None),
    'T27+VIXPRE_common': (C.T27 + C.VIXPRE, '2017-01-01', None),
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'lgb'
    panel = C.load_panel()
    tf = C.test_frame(panel)
    os.makedirs(C.RES, exist_ok=True)
    rows, by_year = [], []
    for name, (feats, start, req) in CONFIGS.items():
        print(f'[{which}] {name}', flush=True)
        p, thr = C.run_config(panel, feats, which, start, req, verbose=False)
        C.save_pred(f'{which}_{name}', p, thr)
        ev = C.evaluate(tf, p, thr)
        b = ev.pop('by_year')
        b.insert(0, 'config', name)
        b.insert(0, 'model', which)
        by_year.append(b)
        rows.append({'model': which, 'config': name, **ev})
    summ = pd.DataFrame(rows)
    pd.concat(by_year).to_csv(os.path.join(C.RES, f'baseline_by_year_{which}.csv'), index=False)
    summ.to_csv(os.path.join(C.RES, f'baseline_summary_{which}.csv'), index=False)
    cols = ['config', 'auroc_yavg', 'auroc_pooled', 'perday_yavg', 'accuracy_yavg', 'macro_f1_yavg', 'pr_auc_yavg',
            'auroc_on_yavg', 'auroc_id_yavg']
    print(summ[cols].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
