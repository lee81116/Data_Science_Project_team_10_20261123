# 前瞻測試與文獻對照（2026-11-22 下載並執行）

Kaggle 資料止於 2026-02-18。之後的行情由 Yahoo Finance 另行下載（外部資料），模型只用 2026-02-17 以前的資料訓練一次，不再重新擬合。
預期在下載前寫下（最後一節）。CI 為配對 20 日區塊 bootstrap 的 95% 區間；holdout 只有 10 個區塊，區間偏粗。

## 下載與對帳

```json
{
  "downloaded_on": "2026-11-22 15:53 UTC",
  "tickers_requested": 100,
  "tickers_downloaded": 100,
  "tickers_missing": [],
  "kaggle_last_date": "2026-02-18",
  "yahoo_last_date": "2026-11-20",
  "tickers_not_reaching_last_date": [
    "EA",
    "WBD"
  ],
  "splits_after_2026_02_18": {
    "BKNG": [
      "2026-04-06"
    ],
    "CRWD": [
      "2026-07-02"
    ],
    "HON": [
      "2026-06-29"
    ],
    "KLAC": [
      "2026-06-12"
    ],
    "MNST": [
      "2026-08-11"
    ]
  },
  "new_rows_after_kaggle_end": 18818,
  "new_trading_days": 192,
  "overlap_rows_compared": 76559,
  "overlap_return_abs_diff_median": 2.656747794427261e-06,
  "overlap_return_abs_diff_p999": 0.029869853750290512,
  "overlap_return_abs_diff_max": 0.104954584597585,
  "overlap_share_abs_diff_above_1e-4": 0.057080160399169265,
  "overlap_return_corr": 0.997178827834767,
  "vix_old_last_date": "11/17/2026",
  "vix_new_last_date": "11/20/2026",
  "vix_overlap_days": 9318,
  "vix_overlap_max_abs_diff_open": 0.0,
  "vix_overlap_max_abs_diff_close": 0.0,
  "worst_overlap_rows": [
    {
      "Ticker": "AZN",
      "Date": "2025-02-06",
      "ret_kaggle": 0.12497154564079227,
      "ret_yahoo": 0.02001696104320727
    },
    {
      "Ticker": "AZN",
      "Date": "2024-05-29",
      "ret_kaggle": -0.07758676613687976,
      "ret_yahoo": -0.00704774707514777
    },
    {
      "Ticker": "AZN",
      "Date": "2025-02-05",
      "ret_kaggle": -0.034859025997803106,
      "ret_yahoo": 0.02871230267920133
    },
    {
      "Ticker": "AZN",
      "Date": "2024-07-08",
      "ret_kaggle": 0.05580141059327892,
      "ret_yahoo": -0.0027152750997012642
    },
    {
      "Ticker": "AZN",
      "Date": "2025-03-24",
      "ret_kaggle": -0.06524022053032308,
      "ret_yahoo": -0.011210571396959468
    },
    {
      "Ticker": "AZN",
      "Date": "2025-08-25",
      "ret_kaggle": 0.0374944299446176,
      "ret_yahoo": -0.01617885791066298
    },
    {
      "Ticker": "AZN",
      "Date": "2025-04-09",
      "ret_kaggle": -0.024488554262681594,
      "ret_yahoo": 0.028659514561138577
    },
    {
      "Ticker": "AZN",
      "Date": "2025-03-10",
      "ret_kaggle": 0.03332883551477539,
      "ret_yahoo": -0.019483884008168806
    }
  ]
}
```

重建特徵與既有面板在重疊期間（2025-03 至 2026-02）的比對：

| group | feature | n | corr | median_abs_diff | p99_abs_diff | sd_kaggle |
| --- | --- | --- | --- | --- | --- | --- |
| all tickers except AZN | return_1d | 23474 | 0.999999 | 0.000000 | 0.000160 | 0.026829 |
| all tickers except AZN | return_5d | 23474 | 1.000000 | 0.000000 | 0.000161 | 0.057777 |
| all tickers except AZN | return_10d | 23474 | 1.000000 | 0.000000 | 0.000164 | 0.081703 |
| all tickers except AZN | return_20d | 23474 | 1.000000 | 0.000001 | 0.000166 | 0.116689 |
| all tickers except AZN | sma_ratio_5 | 23474 | 0.999999 | 0.000000 | 0.000099 | 0.028774 |
| all tickers except AZN | sma_ratio_10 | 23474 | 1.000000 | 0.000000 | 0.000106 | 0.043670 |
| all tickers except AZN | sma_ratio_20 | 23474 | 1.000000 | 0.000001 | 0.000113 | 0.064224 |
| all tickers except AZN | sma_ratio_50 | 23474 | 1.000000 | 0.000001 | 0.000113 | 0.104005 |
| all tickers except AZN | sma_ratio_200 | 23474 | 1.000000 | 0.000001 | 0.000112 | 0.200175 |
| all tickers except AZN | ema_ratio_12 | 23474 | 1.000000 | 0.000000 | 0.000098 | 0.041262 |
| all tickers except AZN | ema_ratio_26 | 23474 | 1.000000 | 0.000001 | 0.000107 | 0.063398 |
| all tickers except AZN | macd_hist | 23474 | 1.000000 | 0.000000 | 0.000010 | 0.008931 |
| all tickers except AZN | rsi_14 | 23474 | 1.000000 | 0.000317 | 0.045235 | 12.676867 |
| all tickers except AZN | bb_position | 23474 | 1.000000 | 0.000008 | 0.002834 | 0.657626 |
| all tickers except AZN | volatility_10d | 23474 | 1.000000 | 0.000002 | 0.000995 | 0.232395 |
| all tickers except AZN | volatility_20d | 23474 | 1.000000 | 0.000005 | 0.000745 | 0.208195 |
| all tickers except AZN | volatility_60d | 23474 | 1.000000 | 0.000007 | 0.000425 | 0.180338 |
| all tickers except AZN | volume_sma_ratio_10 | 23474 | 1.000000 | 0.000000 | 0.000000 | 0.420042 |
| all tickers except AZN | volume_sma_ratio_20 | 23474 | 1.000000 | 0.000000 | 0.000000 | 0.489510 |
| all tickers except AZN | daily_range | 23474 | 0.999999 | 0.000000 | 0.000140 | 0.019831 |
| all tickers except AZN | avg_range_10d | 23474 | 1.000000 | 0.000000 | 0.000039 | 0.013885 |
| all tickers except AZN | high_low_ratio | 23474 | 0.999999 | 0.000000 | 0.000144 | 0.021148 |
| all tickers except AZN | momentum_10d | 23474 | 1.000000 | 0.000000 | 0.000164 | 0.081703 |
| all tickers except AZN | momentum_20d | 23474 | 1.000000 | 0.000001 | 0.000166 | 0.116689 |
| all tickers except AZN | roc_5 | 23474 | 1.000000 | 0.000000 | 0.000161 | 0.057777 |
| all tickers except AZN | roc_10 | 23474 | 1.000000 | 0.000000 | 0.000164 | 0.081703 |
| all tickers except AZN | atr_14 | 23474 | 1.000000 | 0.000001 | 0.000025 | 0.014741 |
| all tickers except AZN | rho | 23474 | 0.999938 | 0.001471 | 0.007430 | 0.198389 |
| all tickers except AZN | gap_z60 | 23474 | 1.000000 | 0.000000 | 0.000000 | 1.063050 |
| all tickers except AZN | gap_next | 23474 | 1.000000 | 0.000000 | 0.000000 | 0.049317 |
| all tickers except AZN | fut_ret | 23474 | 0.999998 | 0.000000 | 0.000160 | 0.026799 |
| all tickers except AZN | on_ret | 23474 | 0.999994 | 0.000000 | 0.000182 | 0.016249 |
| all tickers except AZN | id_ret | 23474 | 1.000000 | 0.000000 | 0.000116 | 0.021900 |
| all tickers except AZN | label y agreement | 23474 | 1.000000 |  |  |  |
| AZN only | return_1d | 51 | 0.611879 | 0.013013 | 0.052290 | 0.026532 |
| AZN only | return_5d | 51 | 0.906552 | 0.016044 | 0.065562 | 0.066447 |
| AZN only | return_10d | 51 | 0.935540 | 0.018602 | 0.055841 | 0.074451 |
| AZN only | return_20d | 51 | 0.967054 | 0.009813 | 0.057822 | 0.072578 |
| AZN only | sma_ratio_5 | 51 | 0.817827 | 0.007385 | 0.039242 | 0.027364 |
| AZN only | sma_ratio_10 | 51 | 0.922839 | 0.010365 | 0.039939 | 0.044126 |
| AZN only | sma_ratio_20 | 51 | 0.958079 | 0.007658 | 0.041713 | 0.056679 |
| AZN only | sma_ratio_50 | 51 | 0.968901 | 0.007293 | 0.044379 | 0.066271 |
| AZN only | sma_ratio_200 | 51 | 0.990201 | 0.006723 | 0.053591 | 0.129327 |
| AZN only | ema_ratio_12 | 51 | 0.917670 | 0.007774 | 0.036942 | 0.038040 |
| AZN only | ema_ratio_26 | 51 | 0.949272 | 0.007362 | 0.041281 | 0.050782 |
| AZN only | macd_hist | 51 | 0.978281 | 0.001510 | 0.004553 | 0.008437 |
| AZN only | rsi_14 | 51 | 0.946696 | 2.051044 | 11.001745 | 8.858449 |
| AZN only | bb_position | 51 | 0.926288 | 0.118926 | 0.758955 | 0.609306 |
| AZN only | volatility_10d | 51 | 0.356267 | 0.126590 | 0.369601 | 0.083079 |
| AZN only | volatility_20d | 51 | 0.467826 | 0.178285 | 0.221565 | 0.065930 |
| AZN only | volatility_60d | 51 | 0.492599 | 0.116857 | 0.283444 | 0.090000 |
| AZN only | volume_sma_ratio_10 | 51 | 0.078199 | 0.420588 | 6.353405 | 1.540829 |
| AZN only | volume_sma_ratio_20 | 51 | -0.083617 | 0.507881 | 13.759702 | 3.068595 |
| AZN only | daily_range | 51 | 0.349159 | 0.010873 | 0.043142 | 0.015430 |
| AZN only | avg_range_10d | 51 | 0.463577 | 0.004217 | 0.018941 | 0.007472 |
| AZN only | high_low_ratio | 51 | 0.364018 | 0.011213 | 0.046179 | 0.016177 |
| AZN only | momentum_10d | 51 | 0.935540 | 0.018602 | 0.055841 | 0.074451 |
| AZN only | momentum_20d | 51 | 0.967054 | 0.009813 | 0.057822 | 0.072578 |
| AZN only | roc_5 | 51 | 0.906552 | 0.016044 | 0.065562 | 0.066447 |
| AZN only | roc_10 | 51 | 0.935540 | 0.018602 | 0.055841 | 0.074451 |
| AZN only | atr_14 | 51 | 0.670380 | 0.010353 | 0.017047 | 0.004765 |
| AZN only | rho | 51 | 0.849871 | 0.217309 | 0.255460 | 0.038988 |
| AZN only | gap_z60 | 51 | 1.000000 | 0.000000 | 0.000000 | 1.172956 |
| AZN only | gap_next | 51 | 1.000000 | 0.000000 | 0.000000 | 0.052360 |
| AZN only | fut_ret | 51 | 0.627703 | 0.011107 | 0.052290 | 0.026391 |
| AZN only | on_ret | 51 | 0.634220 | 0.011496 | 0.051576 | 0.029721 |
| AZN only | id_ret | 51 | 0.490399 | 0.008196 | 0.058755 | 0.021319 |
| AZN only | label y agreement | 51 | 0.823529 |  |  |  |

## Holdout 結果

```json
{
  "holdout_first_day": "2026-02-18",
  "holdout_last_day": "2026-11-19",
  "days": 192,
  "bootstrap_blocks": 10,
  "rows": 18816,
  "tickers": 98,
  "up_rate": 0.49872449040412903,
  "train_up_rate": 0.5142397773755066,
  "market_return_sum_pct": 11.536767959594727,
  "days_gap_abs_ge_1p5": 25,
  "train_last_day": "2026-02-13",
  "M1_perday_without_AZN": 0.5479107743501717,
  "M1_auroc_without_AZN": 0.5428490941543773,
  "krauss_perday_auc": 0.504040200263779,
  "perday_lo": 0.49034310057759284,
  "perday_hi": 0.5173550963401794,
  "k10_accuracy": 0.4963541626930237,
  "acc_lo": 0.47201088070869446,
  "acc_hi": 0.5218750238418579,
  "k10_long_short_daily_return_pct": -0.21545179188251495,
  "ls_lo": -0.6410147703718394,
  "ls_hi": 0.14039978501386924
}
```

| model | auroc_y | auroc_y_lo | auroc_y_hi | auroc_y_on | auroc_y_on_lo | auroc_y_on_hi | auroc_y_id | auroc_y_id_lo | auroc_y_id_hi | perday | perday_lo | perday_hi | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LGB T27 (16:00) | 0.5034 | 0.4880 | 0.5192 | 0.5179 | 0.5078 | 0.5284 | 0.4851 | 0.4722 | 0.4988 | 0.5007 | 0.4848 | 0.5166 | 0.5026 |
| LGB T27+VIXPRE | 0.5236 | 0.5128 | 0.5386 | 0.5713 | 0.5391 | 0.6018 | 0.4862 | 0.4645 | 0.5120 | 0.5038 | 0.4838 | 0.5238 | 0.5196 |
| LGB T27 (16:00), 2017+ | 0.4997 | 0.4814 | 0.5177 | 0.5154 | 0.5002 | 0.5308 | 0.4811 | 0.4689 | 0.4935 | 0.4997 | 0.4811 | 0.5170 | 0.5005 |
| LGB T27+VIXPRE, 2017+ | 0.5245 | 0.5130 | 0.5407 | 0.5713 | 0.5408 | 0.5998 | 0.4870 | 0.4649 | 0.5135 | 0.5063 | 0.4839 | 0.5290 | 0.5212 |
| M1 rho x gap | 0.5427 | 0.5278 | 0.5618 | 0.5903 | 0.5651 | 0.6193 | 0.4946 | 0.4708 | 0.5227 | 0.5471 | 0.5243 | 0.5729 | 0.5251 |
| D1 rho x gap + selection | 0.5406 | 0.5268 | 0.5570 | 0.5937 | 0.5716 | 0.6180 | 0.4902 | 0.4679 | 0.5151 | 0.5334 | 0.5085 | 0.5620 | 0.5218 |
| selection score only (16:00) | 0.5006 | 0.4862 | 0.5143 | 0.5135 | 0.4980 | 0.5278 | 0.4865 | 0.4719 | 0.4991 | 0.4985 | 0.4836 | 0.5125 |  |
| zero-parameter: -VIX gap | 0.5311 | 0.5166 | 0.5489 | 0.5670 | 0.5336 | 0.6009 | 0.4971 | 0.4723 | 0.5282 |  |  |  |  |
| zero-parameter: -market return(t) | 0.5078 | 0.4963 | 0.5207 | 0.5159 | 0.4893 | 0.5482 | 0.4958 | 0.4750 | 0.5155 |  |  |  |  |

| a | b | metric | delta | lo | hi | p |
| --- | --- | --- | --- | --- | --- | --- |
| LGB T27 (16:00) | LGB T27+VIXPRE | pooled | 0.0202 | -0.0015 | 0.0432 | 0.0720 |
| LGB T27 (16:00), 2017+ | LGB T27+VIXPRE, 2017+ | pooled | 0.0248 | 0.0004 | 0.0509 | 0.0458 |
| LGB T27+VIXPRE, 2017+ | M1 rho x gap | perday | 0.0408 | 0.0132 | 0.0631 | 0.0060 |
| LGB T27+VIXPRE, 2017+ | M1 rho x gap | pooled | 0.0182 | 0.0086 | 0.0279 | 0.0002 |
| LGB T27+VIXPRE, 2017+ | D1 rho x gap + selection | perday | 0.0271 | 0.0067 | 0.0466 | 0.0120 |
| LGB T27+VIXPRE, 2017+ | D1 rho x gap + selection | pooled | 0.0161 | 0.0064 | 0.0274 | 0.0012 |
| M1 rho x gap | D1 rho x gap + selection | perday | -0.0137 | -0.0306 | 0.0017 | 0.0840 |
| LGB T27+VIXPRE | M1 rho x gap | perday | 0.0433 | 0.0181 | 0.0659 | 0.0020 |

| model | bucket | days | perday_auc | lo | hi |
| --- | --- | --- | --- | --- | --- |
| M1 rho x gap | |g|<0.5 | 104 | 0.5261 | 0.4965 | 0.5528 |
| M1 rho x gap | 0.5<=|g|<1.5 | 63 | 0.5743 | 0.5340 | 0.6094 |
| M1 rho x gap | |g|>=1.5 | 25 | 0.5661 | 0.4845 | 0.6459 |
| D1 rho x gap + selection | |g|<0.5 | 104 | 0.5137 | 0.4864 | 0.5367 |
| D1 rho x gap + selection | 0.5<=|g|<1.5 | 63 | 0.5543 | 0.5150 | 0.5919 |
| D1 rho x gap + selection | |g|>=1.5 | 25 | 0.5625 | 0.4790 | 0.6423 |
| LGB T27+VIXPRE, 2017+ | |g|<0.5 | 104 | 0.4946 | 0.4692 | 0.5166 |
| LGB T27+VIXPRE, 2017+ | 0.5<=|g|<1.5 | 63 | 0.5129 | 0.4799 | 0.5464 |
| LGB T27+VIXPRE, 2017+ | |g|>=1.5 | 25 | 0.5381 | 0.4648 | 0.5966 |

換算式（holdout）：

| label | auc_market | mkt_lo | mkt_hi | rho_mean | auc_stock_predicted | auc_stock_observed | obs_lo | obs_hi | obs_minus_pred | diff_lo | diff_hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| close-to-close | 0.6445 | 0.5809 | 0.7086 | 0.3059 | 0.5439 | 0.5311 | 0.5166 | 0.5489 | -0.0128 | -0.0339 | 0.0071 |
| overnight | 0.7813 | 0.7233 | 0.8397 | 0.2765 | 0.5757 | 0.5670 | 0.5336 | 0.6009 | -0.0087 | -0.0437 | 0.0256 |
| intraday | 0.5431 | 0.4528 | 0.6512 | 0.3267 | 0.5141 | 0.4971 | 0.4723 | 0.5282 | -0.0170 | -0.0394 | 0.0041 |

換算式（逐期間）：

| period | holdout | days | auc_market | rho_mean | auc_stock_predicted | auc_stock_observed | obs_minus_pred |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2018 | False | 251 | 0.6684 | 0.6036 | 0.6009 | 0.6194 | 0.0184 |
| 2019 | False | 252 | 0.6467 | 0.5275 | 0.5769 | 0.5891 | 0.0122 |
| 2020 | False | 253 | 0.6261 | 0.7030 | 0.5884 | 0.6006 | 0.0122 |
| 2021 | False | 252 | 0.6256 | 0.5074 | 0.5635 | 0.5772 | 0.0137 |
| 2022 | False | 251 | 0.5811 | 0.6704 | 0.5543 | 0.5570 | 0.0027 |
| 2023 | False | 250 | 0.6311 | 0.5014 | 0.5654 | 0.5561 | -0.0093 |
| 2024 | False | 252 | 0.5774 | 0.4150 | 0.5321 | 0.5427 | 0.0106 |
| 2025 | False | 250 | 0.6400 | 0.5116 | 0.5712 | 0.5695 | -0.0018 |
| 2026-02 to 11 (holdout) | True | 192 | 0.6445 | 0.3059 | 0.5439 | 0.5311 | -0.0128 |

## Walk-forward 的最後兩年（既有資料）

| period | days | model | auroc | auroc_lo | auroc_hi | perday | perday_lo | perday_hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | 250 | LGB T27 (16:00) | 0.5245 | 0.5144 | 0.5354 | 0.5146 | 0.5049 | 0.5234 |
| 2025 | 250 | LGB T27+VIXPRE | 0.5738 | 0.5402 | 0.6099 | 0.5180 | 0.5089 | 0.5275 |
| 2025 | 250 | LGB T27+VIXPRE, 2017+ | 0.5714 | 0.5388 | 0.6079 | 0.5142 | 0.5045 | 0.5239 |
| 2025 | 250 | M1 rho x gap | 0.5740 | 0.5411 | 0.6082 | 0.5485 | 0.5254 | 0.5729 |
| 2025 | 250 | D1 rho x gap + selection | 0.5735 | 0.5424 | 0.6061 | 0.5446 | 0.5229 | 0.5679 |
| 2025 + 2026 (31 days) | 281 | LGB T27 (16:00) | 0.5212 | 0.5114 | 0.5321 | 0.5113 | 0.5016 | 0.5207 |
| 2025 + 2026 (31 days) | 281 | LGB T27+VIXPRE | 0.5679 | 0.5369 | 0.6024 | 0.5131 | 0.5022 | 0.5241 |
| 2025 + 2026 (31 days) | 281 | LGB T27+VIXPRE, 2017+ | 0.5660 | 0.5361 | 0.6003 | 0.5105 | 0.5005 | 0.5205 |
| 2025 + 2026 (31 days) | 281 | M1 rho x gap | 0.5692 | 0.5404 | 0.6005 | 0.5443 | 0.5233 | 0.5669 |
| 2025 + 2026 (31 days) | 281 | D1 rho x gap + selection | 0.5682 | 0.5405 | 0.5982 | 0.5404 | 0.5211 | 0.5619 |
| 2018-2024 | 1761 | LGB T27 (16:00) | 0.5115 | 0.5036 | 0.5197 | 0.5134 | 0.5082 | 0.5187 |
| 2018-2024 | 1761 | LGB T27+VIXPRE | 0.5718 | 0.5577 | 0.5859 | 0.5143 | 0.5085 | 0.5199 |
| 2018-2024 | 1761 | LGB T27+VIXPRE, 2017+ | 0.5700 | 0.5553 | 0.5846 | 0.5089 | 0.5032 | 0.5147 |
| 2018-2024 | 1761 | M1 rho x gap | 0.5762 | 0.5617 | 0.5907 | 0.5304 | 0.5222 | 0.5387 |
| 2018-2024 | 1761 | D1 rho x gap + selection | 0.5769 | 0.5627 | 0.5912 | 0.5350 | 0.5277 | 0.5424 |

| period | a | b | metric | delta | lo | hi | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | LGB T27 (16:00) | LGB T27+VIXPRE | pooled | 0.0493 | 0.0144 | 0.0831 | 0.0060 |
| 2025 | LGB T27+VIXPRE, 2017+ | M1 rho x gap | pooled | 0.0026 | -0.0083 | 0.0154 | 0.6733 |
| 2025 | LGB T27+VIXPRE, 2017+ | M1 rho x gap | perday | 0.0343 | 0.0129 | 0.0559 | 0.0030 |
| 2025 + 2026 (31 days) | LGB T27 (16:00) | LGB T27+VIXPRE | pooled | 0.0467 | 0.0157 | 0.0782 | 0.0038 |
| 2025 + 2026 (31 days) | LGB T27+VIXPRE, 2017+ | M1 rho x gap | pooled | 0.0033 | -0.0062 | 0.0137 | 0.5155 |
| 2025 + 2026 (31 days) | LGB T27+VIXPRE, 2017+ | M1 rho x gap | perday | 0.0338 | 0.0149 | 0.0534 | 0.0016 |
| 2018-2024 | LGB T27 (16:00) | LGB T27+VIXPRE | pooled | 0.0603 | 0.0472 | 0.0732 | 0.0002 |
| 2018-2024 | LGB T27+VIXPRE, 2017+ | M1 rho x gap | pooled | 0.0062 | -0.0016 | 0.0142 | 0.1212 |
| 2018-2024 | LGB T27+VIXPRE, 2017+ | M1 rho x gap | perday | 0.0215 | 0.0124 | 0.0305 | 0.0002 |

## Krauss 等人（2017）協定，依其子期間切分（本組 LightGBM；報酬單位 %）

| k | window | days | stocks_per_day | accuracy | mean_daily_ls_pct | ann_before_cost_pct | ann_after_cost_pct | acc_lo | acc_hi | daily_lo_pct | daily_hi_pct | ann_before_lo_pct | ann_before_hi_pct | ann_after_lo_pct | ann_after_hi_pct | perday_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 2005-01 to 2008-08 | 922 | 51.695 | 0.512 | 0.011 | 1.778 | -38.298 | 0.505 | 0.518 | -0.043 | 0.066 | -11.117 | 16.705 | -46.131 | -29.230 | 0.510 |
| 10 | 2008-09 to 2009-12 | 337 | 51.151 | 0.508 | 0.132 | 35.764 | -17.650 | 0.497 | 0.518 | -0.013 | 0.300 | -5.484 | 105.833 | -42.712 | 24.956 | 0.507 |
| 10 | 2010-01 to 2015-10 | 1468 | 67.252 | 0.516 | 0.091 | 24.484 | -24.503 | 0.510 | 0.523 | 0.046 | 0.133 | 11.033 | 38.105 | -32.676 | -16.224 | 0.512 |
| 10 | 2015-11 to 2017-12 | 545 | 79.580 | 0.513 | 0.030 | 6.584 | -35.379 | 0.503 | 0.523 | -0.025 | 0.084 | -7.282 | 22.459 | -43.802 | -25.733 | 0.509 |
| 10 | 2018-01 to 2026-02 | 2042 | 92.371 | 0.511 | 0.016 | 0.607 | -39.013 | 0.505 | 0.517 | -0.055 | 0.091 | -15.547 | 21.200 | -48.823 | -26.502 | 0.509 |
| 2 | 2005-01 to 2008-08 | 922 | 51.695 | 0.509 | 0.015 | -2.750 | -41.055 | 0.493 | 0.526 | -0.130 | 0.164 | -32.134 | 41.502 | -58.895 | -14.170 |  |
| 2 | 2008-09 to 2009-12 | 337 | 51.151 | 0.517 | 0.268 | 77.295 | 7.584 | 0.492 | 0.543 | 0.016 | 0.522 | -5.799 | 233.531 | -42.912 | 102.652 |  |
| 2 | 2010-01 to 2015-10 | 1468 | 67.252 | 0.531 | 0.182 | 49.298 | -9.429 | 0.519 | 0.542 | 0.093 | 0.268 | 19.419 | 85.419 | -27.589 | 12.534 |  |
| 2 | 2015-11 to 2017-12 | 545 | 79.580 | 0.510 | -0.015 | -8.680 | -44.655 | 0.487 | 0.531 | -0.215 | 0.163 | -45.454 | 43.859 | -66.978 | -12.733 |  |
| 2 | 2018-01 to 2026-02 | 2042 | 92.371 | 0.510 | -0.058 | -24.548 | -54.298 | 0.499 | 0.522 | -0.209 | 0.097 | -48.443 | 10.877 | -68.796 | -32.788 |  |

## 下載前寫下的預期

```json
{
  "written": "2026-11-22 (Taipei), before any price dated after 2026-02-18 was downloaded and before any holdout metric was computed",
  "holdout_period": "2026-02-18 to the last labelled day available (about 190 trading days); models are trained once on data up to 2026-02-17 and not refitted",
  "basis": "yearly walk-forward results 2018-2025 from this pipeline",
  "expectations": {
    "LGB_T27_pooled_auroc": "0.50 to 0.53 (yearly range so far 0.50 to 0.54); CI likely to include 0.5",
    "LGB_T27+VIXPRE_pooled_auroc": "0.54 to 0.60 (yearly range so far 0.53 to 0.61)",
    "timing_effect_T27_to_T27+VIXPRE": "+0.03 to +0.08, CI excludes 0",
    "M1_pooled_auroc": "0.54 to 0.60",
    "M1_perday_auc": "0.51 to 0.55, point estimate above 0.5 (about 85% confident); with about 190 days the CI half-width is near 0.02, so the CI may include 0.5",
    "M1_minus_LGB_perday": "+0.01 to +0.04 (positive in 7 of 8 full test years so far)",
    "overnight_vs_intraday": "T27+VIXPRE overnight AUROC 0.60 to 0.68, intraday AUROC 0.49 to 0.52",
    "conversion_law": "predicted stock-level AUROC of the VIX gap within 0.02 of the observed value",
    "krauss_features": "per-day AUC 0.50 to 0.52, k=10 accuracy 50% to 52%"
  },
  "what_would_count_as_failure": "T27+VIXPRE minus T27 not positive, or M1 per-day AUC below 0.5: the main claims would not replicate prospectively and would be reported that way"
}
```
