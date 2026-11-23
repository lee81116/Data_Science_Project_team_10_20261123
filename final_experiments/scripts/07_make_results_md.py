"""Collect every result CSV into RESULTS.md (tables only; interpretation lives in the plan document)."""
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
    return pd.read_csv(os.path.join(C.RES, name))


def main():
    acc = json.load(open(os.path.join(C.RES, 'panel_row_accounting.json')))
    chk = json.load(open(os.path.join(C.RES, 'spec4_checks.json')))
    out = ['# 實驗結果彙整（2026-11-22 重建）', '',
           '由原始 OHLCV 與 Cboe VIX 日線重建面板後執行。協定依 `reports_docs/EXPERIMENT_SETUP_REPORT.md`；只跑 global 模式。',
           '所有 CI 為配對 20 日區塊 bootstrap（10,000 次）的 95% 區間。解讀與結論見計劃書文件的「結果」分頁。', '']
    out += ['## 0. 面板列數對帳', '', md(pd.DataFrame([{'項目': k, '本次重建': v, '組員報告': acc['reported'].get(rk, '')} for k, v, rk in [
        ('原始列數', acc['raw_rows'], 'raw_rows'), ('指標缺失或成交量為 0', acc['drop_missing_indicator_or_zero_volume'], 'drop_missing_or_zero_volume'),
        ('無次日標籤', acc['drop_no_next_day_label'], 'drop_no_label'), ('Adj Close < 5', acc['drop_adj_close_below_5'], 'drop_below_5'),
        ('早於 2002-01-01', acc['drop_before_2002'], ''), ('分析面板', acc['panel_rows'], 'panel_rows'), ('測試列 2018–2026', acc['test_rows_2018_2026'], 'test_rows')]])), '']
    b = r('baseline_summary_lgb.csv'); bl = r('baseline_summary_lr.csv')
    cols = ['model', 'config', 'auroc_yavg', 'auroc_pooled', 'perday_yavg', 'pr_auc_yavg', 'auroc_on_yavg', 'auroc_id_yavg', 'accuracy_yavg', 'macro_f1_yavg']
    out += ['## 1. 基線重現（global）', '', md(pd.concat([b, bl])[cols]), '']
    by = r('baseline_by_year_lgb.csv').pivot(index='year', columns='config', values='auroc')[['T27', 'VIXPRE', 'T27+VIXPRE']].reset_index()
    out += ['逐年 AUROC（LightGBM）：', '', md(by), '']
    for m in ('lgb', 'lr'):
        out += [f'## 2. Spec 1 時間點對照（{m}）', '', md(r(f'spec1_summary_{m}.csv')[['config', 'auroc_pooled', 'auroc_yavg', 'perday_pooled', 'auroc_on_pooled', 'auroc_id_pooled']]), '',
                md(r(f'spec1_deltas_{m}.csv')), '', md(r(f'spec1_levels_{m}.csv')), '']
    out += ['## 3. Spec 2 擇時 × 選股', '', md(r('spec2_metrics.csv')), '', md(r('spec2_deltas.csv')), '', md(r('spec2_buckets.csv')), '',
            md(r('spec2_rho_stats.csv').rename(columns={'Unnamed: 0': 'stat', '0': 'value'})), '']
    s3 = r('spec3_metrics.csv')
    for c in [c for c in s3.columns if 'ls_ret' in c]:
        s3[c] = s3[c] * 100
    out += ['## 4. Spec 3 文獻可比基準（ls_ret 單位為每日 %）', '', md(s3), '', md(r('spec3_deltas.csv')), '', md(r('spec3_perday_cs_by_year.csv')), '',
            md(r('spec3_step3_coefs.csv')), '']
    sd = r('spec4_seeds.csv').groupby('config')[['auroc_yavg', 'auroc_pooled', 'perday_yavg', 'auroc_on_yavg', 'auroc_id_yavg']].agg(['mean', 'std'])
    sd.columns = [f'{a}_{b}' for a, b in sd.columns]
    out += ['## 5. Spec 4 檢查', '', '多 seed（42–46）：', '', md(sd.reset_index()), '', '分類別表現（逐年平均）：', '', md(r('spec4_classwise.csv')), '',
            '與 Progress 協定重疊的年份：', '', md(r('spec4_progress_overlap.csv')), '', '零參數的日期層級分數：', '', md(r('spec4_zero_param_baselines.csv')), '',
            '各段標籤的逐日 AUC：', '', md(r('spec4_perday_by_segment.csv')), '', 'VIX OPEN 缺口與個股隔夜報酬的相關：', '',
            md(r('spec4_vix_open_periods.csv').rename(columns={'Unnamed: 0': 'period'})), '', '其他：', '', '```json', json.dumps(chk, indent=2), '```', '']
    open(os.path.join(C.ROOT, 'RESULTS.md'), 'w', encoding='utf-8').write('\n'.join(out))
    print('RESULTS.md written,', len(out), 'lines')


if __name__ == '__main__':
    main()
