"""Collect the supplementary result CSVs into RESULTS_SUPP.md (tables only; interpretation lives in the plan document)."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def md(df, floatfmt='{:.4f}'):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: '' if pd.isna(v) else floatfmt.format(v))
    lines = ['| ' + ' | '.join(map(str, df.columns)) + ' |', '| ' + ' | '.join(['---'] * len(df.columns)) + ' |']
    lines += ['| ' + ' | '.join(map(str, r)) + ' |' for r in df.itertuples(index=False)]
    return '\n'.join(lines)


def r(name):
    path = os.path.join(C.RES, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{name} missing: run scripts 08 to 11 first')
    return pd.read_csv(path)


def main():
    exp = json.load(open(os.path.join(C.RES, 'supp_expectations.json')))
    out = ['# 補強實驗結果彙整（2026-11-22）', '',
           '針對四個弱點各做一項補強，全部使用既有資料。CI 為配對 20 日區塊 bootstrap 的 95% 區間（10,000 次；年代分析 5,000 次）。',
           '解讀見計劃書文件的「補強結果」分頁。事前寫下的預期在最後一節。', '']
    out += ['## S5 管線統一：重建 stock 模式', '', '重建值與組員報告值（reported_*，僅 LightGBM 有報告值）、同一批列上的 global 對照：', '', md(r('supp5_stock_summary.csv')), '',
            'global 減 stock（配對）；最後兩列是 M1 減 stock VIXPRE：', '', md(r('supp5_stock_vs_global.csv')), '', '依缺口大小分組的逐日 AUC：', '', md(r('supp5_stock_buckets.csv')), '']
    out += ['## S6 分解式模型（03:15 預測時間點）', '', md(r('supp6_metrics.csv')), '', md(r('supp6_deltas.csv')), '', md(r('supp6_buckets.csv')), '',
            '組合器係數（近乎不正則化、原始單位）：', '', md(r('supp6_combiner_coefs.csv')), '']
    out += ['## S6b 同一分解放到 16:00 預測時間點', '', md(r('supp6b_metrics.csv')), '', md(r('supp6b_deltas.csv')), '']
    out += ['## S7 結構檢驗', '', '曝險定義與 placebo（逐日 AUC）：', '', md(r('supp7_exposure.csv')), '',
            '換算式：由市場層級 AUROC 與 rho 預測個股層級 AUROC：', '', md(r('supp7_conversion.csv')), '',
            '達到指定個股層級 AUROC 所需的市場層級 AUROC：', '', md(r('supp7_required_skill.csv')), '',
            '逐日 AUC：實際值與單因子模型隱含值：', '', md(r('supp7_implied_perday.csv')), '',
            '校準：', '', md(r('supp7_calibration.csv'), '{:.5f}'), '', md(r('supp7_calibration_bins.csv')), '']
    bp = os.path.join(C.RES, 'supp7_brier_paired.json')
    if os.path.exists(bp):
        out += ['Brier 配對差（LightGBM 減 M1）：', '', '```json', json.dumps(json.load(open(bp)), indent=2), '```', '']
    out += ['## S8 穩健性', '', 'M1 逐年：', '', md(r('supp8_by_year.csv')), '', '沒看過的股票（一半股票訓練、另一半評估）：', '', md(r('supp8_unseen_stocks.csv')), '',
            '依進入面板的時間分群：', '', md(r('supp8_cohorts.csv')), '', '年代分析（只用 16:00 資訊）：', '', md(r('supp8_era.csv')), '', md(r('supp8_era_deltas.csv')), '',
            md(r('supp8_era_by_year.csv')), '']
    out += ['## 事前預期（執行前寫下）', '', '```json', json.dumps(exp, indent=2, ensure_ascii=False), '```', '']
    open(os.path.join(C.ROOT, 'RESULTS_SUPP.md'), 'w', encoding='utf-8').write('\n'.join(out))
    print('RESULTS_SUPP.md written,', len(out), 'lines')


if __name__ == '__main__':
    main()
