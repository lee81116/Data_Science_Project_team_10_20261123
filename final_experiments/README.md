# final_experiments

2026-11-22 依「繳交前實驗計劃書」的四份 Spec 重建並執行。組員的腳本與中間檔當時沒有拿到，所以面板是從原始資料重建的。

## 與組員結果的差異

- 面板 429,882 列（組員 430,397），測試列 189,237（組員 184,061）。stock 模式要求每個 ticker-year 至少有 120 列訓練資料，符合的測試列是 184,572，與組員只差 0.3%；推測組員對所有模式都套用了類似的最低訓練列數（未向組員確認）。
- global 模式的逐年 AUROC 與組員報告最多差 0.005，平均差在 0.002 內。
- 重建了 T27、VIXPRE、T27+VIXPRE 的 global 與 stock 兩種模式（stock 模式的四項指標與組員報告最多差 0.001）。TALL 的 66 個額外指標，以及 sector、cross 兩種策略沒有重建（缺指標定義與產業對照表）。
- Spec 3 的 Random Forest 比原論文輕量（200 棵、深度 10、每棵抽 20% 樣本）。

## 重新執行

需要 Python 3.10 以上，以及 `pandas numpy scipy scikit-learn lightgbm matplotlib`。

```bash
python3 scripts/00_build_panel.py            # 重建面板 -> data/panel.pkl
python3 scripts/01_baseline.py lgb           # 基線重現（再跑一次 lr）
python3 scripts/02_spec1_fit.py lgb          # Spec 1 配置（再跑一次 lr）
python3 scripts/02_spec1_boot.py lgb         # Spec 1 bootstrap（再跑一次 lr）
python3 scripts/03_spec3_fit.py lgb          # Spec 3（再跑 lr；rf 用 `rf 2018 2019 ...`）
python3 scripts/03_spec3_eval.py
python3 scripts/04_spec2.py                  # Spec 2
python3 scripts/05_spec4_seeds.py T27        # 多 seed（另有 VIXPRE、T27+VIXPRE）
python3 scripts/05_spec4_checks.py
python3 scripts/06_figures.py
python3 scripts/07_make_results_md.py        # 彙整成 RESULTS.md

# 補強實驗（11/22）
python3 scripts/08_supp_stock.py fit lgb     # S5 stock 模式（再跑 fit lr，最後 eval）
python3 scripts/08_supp_stock.py eval
python3 scripts/09_supp_decomp.py fit        # S6 分解式模型（再跑 eval）
python3 scripts/09_supp_decomp.py eval
python3 scripts/09b_supp_close_time.py       # S6b 同一分解放到 16:00
python3 scripts/10_supp_theory.py            # S7 結構檢驗
python3 scripts/11_supp_robust.py main       # S8 逐年、沒看過的股票、分群
python3 scripts/11_supp_robust.py era        # S8 年代分析（2005–2026）
python3 scripts/12_supp_figures.py
python3 scripts/13_make_supp_md.py           # 彙整成 RESULTS_SUPP.md

# 前瞻測試與文獻對照（11/23，需要網路）
python3 scripts/14_holdout_download.py       # 下載 2023 起的 Yahoo 日線與最新 Cboe VIX，並與 Kaggle 檔對帳
python3 scripts/15_holdout_eval.py build     # 用同一套程式重建特徵，與既有面板比對
python3 scripts/15_holdout_eval.py eval      # 模型只訓練一次，套到 2026-02-18 之後
python3 scripts/16_related_work.py           # Krauss 子期間、walk-forward 最後兩年
python3 scripts/17_holdout_report.py         # 逐期間換算式、兩張圖、RESULTS_HOLDOUT.md
```

## 外部資料（前瞻測試）

- Kaggle 檔止於 2026-02-18。`data/holdout/` 是 2026-11-22 另外從 Yahoo Finance 下載的 100 檔日線（2023-01 起）與 Cboe 官方 VIX 日線，只用於前瞻測試，沒有改動既有面板與任何既有結果。
- 與 Kaggle 檔在重疊期間的日報酬差：中位數 3e-6；AZN 例外（兩個來源的報酬相關只有 0.57，哪一邊正確未判定）。
- EA、WBD 在 Yahoo 已無 2026-02 之後的資料，前瞻期間為 98 檔。
- 執行前的預期在 `results/holdout_expectations.json`。

## 資料夾

| 路徑 | 內容 |
| --- | --- |
| `scripts/` | 全部程式；`common.py` 是共用的 walk-forward、指標與 bootstrap |
| `results/` | 每張表的 CSV |
| `preds/` | 各配置的逐列樣本外預測（`.npy`，順序同 `common.test_frame`） |
| `figures/` | 八張圖（英文標籤，可直接放簡報）；S5、S7、S8 是補強實驗的圖，H1、H2 是前瞻測試的圖 |
| `RESULTS.md` | Spec 1–4 所有表格的彙整 |
| `RESULTS_SUPP.md` | 補強實驗（S5–S8）所有表格的彙整，最後一節是執行前寫下的預期 |
| `RESULTS_HOLDOUT.md` | 前瞻測試（2026-02-18 至 2026-11-19）與文獻對照的所有表格 |
| `data/holdout/` | 前瞻測試用的外部資料與重建的面板 |
| `_staging/` | 傳檔時留下的 0 位元組空檔，可以整個刪掉 |

## RQ1 的模型家族（11/23 補跑）

組員的 v2 設計在 RQ1 列了邏輯迴歸、Random Forest、XGBoost、LightGBM，原本只有邏輯迴歸與 LightGBM 的結果。`scripts/18_rq1_models.py` 補上另外兩種，只做 global 模式，超參數固定、沒有調參（設定與預先寫下的預期在 `results/rq1_models_setup.json`）。另外需要 `xgboost`。

```bash
python3 scripts/18_rq1_models.py fit xgb T27          # 另有 T27+VIXPRE；每折結果快取在 preds/rq1_folds/
python3 scripts/18_rq1_models.py fit rf T27           # 約每折 20 至 40 秒，可中斷後續跑
python3 scripts/18_rq1_models.py eval                 # -> results/rq1_models_{metrics,deltas,by_year}.csv
```

| 模型 | T27 合併 AUROC | T27+VIXPRE 合併 AUROC | 增益（95% 區間） |
|---|---|---|---|
| 邏輯迴歸 | 0.5093 | 0.5729 | +0.064 [+0.052, +0.075] |
| Random Forest | 0.5119 | 0.5695 | +0.058 [+0.046, +0.069] |
| XGBoost | 0.5127 | 0.5711 | +0.058 [+0.046, +0.071] |
| LightGBM | 0.5126 | 0.5712 | +0.059 [+0.046, +0.071] |

T27 上對 LightGBM 的配對差：Random Forest −0.0007 [−0.0023, +0.0008]，XGBoost +0.0001 [−0.0005, +0.0007]，邏輯迴歸 −0.0034 [−0.0064, −0.0002]（Holm 校正後 p = 0.24）。XGBoost 與 LightGBM 的設定對齊，兩者幾乎相同，實質上是線性、bagging 樹、boosting 樹三個家族。預先寫下的三項預期都成立。

## 文獻 baseline 重跑：Mukherjee 等（2025）（11/23）

Mukherjee 等（2025，HICSS-58，*Beyond the bell*）的技術指標 baseline 報告 balanced accuracy 56.52%（473 檔 S&P 500、300 個測試日）。`scripts/19_relwork_mukherjee.py` 把同一套做法搬到我們的面板：前一日的 15 個技術指標（論文表 2）→ XGBoost 回歸收盤價 → 預測價和前一日收盤比出方向。

論文沒寫、這裡假設的：每檔一個模型、XGBoost 預設超參數、用該檔之前所有日子訓練；Schaff Trend Cycle 用標準定義配論文表 2 的視窗。我們這邊用還原價、面板的列與標籤。

```bash
python3 scripts/19_relwork_mukherjee.py ind                          # 指標 -> data/mukherjee_ind.pkl
python3 scripts/19_relwork_mukherjee.py fitA                         # 協定 A：每檔最後 300 列（照論文）
python3 scripts/19_relwork_mukherjee.py fitB 2018 2019 ... 2026      # 協定 B：我們的逐年 walk-forward
python3 scripts/19_relwork_mukherjee.py eval                         # -> results/relwork_mukherjee_{summary,per_ticker}.csv
```

| 設定 | Balanced accuracy（95% 區間） | 同一批列上的 LightGBM T27 | 差距 |
|---|---|---|---|
| 論文報告值 | 56.5% | — | — |
| 協定 A：每檔最後 300 列、98 檔、逐檔平均 | 50.9% [50.5, 51.5] | 51.5% [50.7, 52.7] | −0.6 [−2.0, +0.5] 個百分點 |
| 協定 B：逐年 walk-forward、181,508 列、合併 | 49.8% [49.4, 50.2] | 51.0% [50.5, 51.5] | −1.2 [−1.8, −0.6] 個百分點 |
| 協定 B，預測報酬的 AUROC | 0.496 [0.489, 0.502] | 0.513 | −0.017 [−0.026, −0.009] |

論文的 56.5% 在我們的資料上沒有重現。回歸目標是價格水準，樹模型不會外插：協定 A 有 24% 的測試列價格高於訓練期最高價，模型只在 36% 的列預測上漲（實際 51%）。把回歸目標換成次日報酬（`fitA ret`、`fitB ret`，這是我們自己加的對照）也在五成附近（50.0%、50.1%）。較淺的 XGBoost 設定（300 棵、深度 3、學習率 0.05）是 50.7%。

差距的原因無法判斷：股票池、期間、價格是否還原、沒有盤前盤後交易的日子被排除，這些都和論文不同，論文也沒有給超參數與區間。

## 文獻 baseline 重跑：Wang 等（2026）的技術分支（11/23）

Wang 等（2026，Sci Rep 16:17494，FusionLSTM-CNF）在 30 檔 S&P 500 上報告：LSTM baseline accuracy 0.523、AUC 0.567（表 3，輸入沒有說明）；只用技術特徵的分支 0.534、0.578（表 6）。`scripts/20_relwork_wang.py` 把技術分支搬到我們的股票上，訓練、驗證、測試的日期照論文（測試期 2024-03 至 2024-12）。新聞分支、跨資產分支與融合沒有重跑，我們沒有新聞資料。

照論文的：23 個價量與技術輸入、60 日序列、兩層 LSTM（hidden 128、dropout 0.3）、layer norm、AdamW（學習率 0.001）、cosine warm restarts、batch 64、驗證 loss 連續 15 個 epoch 沒有改善就停、gradient clipping 1.0、用訓練期的平均與標準差標準化。
論文沒寫、這裡假設的：均線以外的指標視窗用常見預設；所有股票共用一個模型，輸入逐檔標準化；head 的結構；binary cross-entropy、沒有 label smoothing；warm restart 每 10 個 epoch。這台機器只有 CPU，每次訓練上限設 30 個 epoch（三次都在第 16 到 18 個 epoch 提早停止，上限沒有用到）。

```bash
python3 scripts/20_relwork_wang.py prep            # 需要 torch；序列 -> preds/wang/data.npz（約 650 MB，沒有放進資料夾）
python3 scripts/20_relwork_wang.py fit 42 30       # 一次訓練；本次跑了種子 42、43、44
python3 scripts/20_relwork_wang.py eval            # -> results/relwork_wang_{summary,runs}.csv
```

測試期 2024-03-01 至 2024-12-31，98 檔、211 個交易日、20,618 列：

| 模型 | 合併 AUROC（95% 區間） | Accuracy |
|---|---|---|
| 論文：LSTM baseline（30 檔 S&P 500） | 0.567 | 52.3% |
| 論文：技術分支 | 0.578 | 53.4% |
| 重跑：技術分支 LSTM，驗證 loss 最低的一次 | 0.502 [0.494, 0.511] | 51.3%（每一列都預測上漲） |
| 重跑：三次的平均 | 0.499 [0.491, 0.508] | 51.3% |
| LightGBM，同樣 23 個輸入（最後一天的值）、同樣的訓練期間 | 0.503 [0.490, 0.517] | 50.0% |
| 我們的 LightGBM T27（walk-forward，訓練 2002–2023） | 0.513 [0.497, 0.531] | 51.4% |

三次訓練的測試 AUROC 是 0.497、0.500、0.502，驗證 loss 最低點在第 1、1、3 個 epoch。重跑的 LSTM 對 LightGBM T27 的差距是 −0.011 [−0.034, +0.012]，分不開。論文的 0.567 與 0.578 在我們的資料上沒有重現；只跑了一種設定、三個種子，原因無法判斷。
