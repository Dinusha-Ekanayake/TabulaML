# Benchmark: Random Forest vs gradient boosting

Repeated cross-validation (5 folds x 3 repeats, seed 42); identical folds for every model; TabulaML Automatic-mode preprocessing for every dataset. Error metrics (RMSE) are lower-is-better; ROC-AUC, F1 and R2 are higher-is-better.

## lk_colombo_dengue (real, public (Sri Lanka))

| Model | Metric | Mean | Std | Fit time (s) |
|---|---|---|---|---|
| Random Forest | rmse | 62.3788 | 7.3387 | 0.91 |
| XGBoost | rmse | 64.4065 | 7.8262 | 2.18 |
| LightGBM | rmse | 67.4801 | 6.8945 | 0.94 |
| CatBoost | rmse | 62.6423 | 6.9696 | 1.48 |
| Gradient Boosting | rmse | 64.9620 | 6.4081 | 0.12 |
| Ridge | rmse | 66.8489 | 13.4787 | 0.02 |

## lk_tea_auction_catalogue (real, public (Sri Lanka))

| Model | Metric | Mean | Std | Fit time (s) |
|---|---|---|---|---|
| Random Forest | roc_auc | 0.9614 | 0.0016 | 1.46 |
| XGBoost | roc_auc | 0.9691 | 0.0015 | 5.08 |
| LightGBM | roc_auc | 0.9689 | 0.0014 | 4.34 |
| CatBoost | roc_auc | 0.9736 | 0.0014 | 6.48 |
| Gradient Boosting | roc_auc | 0.9749 | 0.0014 | 11.73 |
| Logistic Regression | roc_auc | 0.9739 | 0.0012 | 0.76 |

## lk_tea_auction_price (real, public (Sri Lanka))

| Model | Metric | Mean | Std | Fit time (s) |
|---|---|---|---|---|
| Random Forest | rmse | 104.7598 | 8.6848 | 5.28 |
| XGBoost | rmse | 99.4603 | 7.9925 | 1.53 |
| LightGBM | rmse | 97.6938 | 6.5641 | 1.38 |
| CatBoost | rmse | 104.3559 | 5.7850 | 2.92 |
| Gradient Boosting | rmse | 146.7556 | 7.8591 | 3.96 |
| Ridge | rmse | 232.9814 | 15.1595 | 0.69 |

## breast_cancer (public benchmark (sklearn))

| Model | Metric | Mean | Std | Fit time (s) |
|---|---|---|---|---|
| Random Forest | roc_auc | 0.9898 | 0.0092 | 0.73 |
| XGBoost | roc_auc | 0.9939 | 0.0049 | 0.43 |
| LightGBM | roc_auc | 0.9936 | 0.0048 | 0.46 |
| CatBoost | roc_auc | 0.9943 | 0.0044 | 2.07 |
| Gradient Boosting | roc_auc | 0.9924 | 0.0052 | 0.48 |
| Logistic Regression | roc_auc | 0.9952 | 0.0045 | 0.04 |

## wine (public benchmark (sklearn))

| Model | Metric | Mean | Std | Fit time (s) |
|---|---|---|---|---|
| Random Forest | roc_auc | 0.9981 | 0.0035 | 0.41 |
| XGBoost | roc_auc | 0.9962 | 0.0047 | 0.67 |
| LightGBM | roc_auc | 0.9968 | 0.0047 | 0.38 |
| CatBoost | roc_auc | 0.9981 | 0.0027 | 1.27 |
| Gradient Boosting | roc_auc | 0.9956 | 0.0055 | 0.27 |
| Logistic Regression | roc_auc | 0.9973 | 0.0032 | 0.02 |

## diabetes (public benchmark (sklearn))

| Model | Metric | Mean | Std | Fit time (s) |
|---|---|---|---|---|
| Random Forest | rmse | 57.8605 | 3.4269 | 0.43 |
| XGBoost | rmse | 60.1238 | 3.8883 | 0.77 |
| LightGBM | rmse | 62.2977 | 3.5008 | 0.31 |
| CatBoost | rmse | 58.7170 | 3.9857 | 2.25 |
| Gradient Boosting | rmse | 58.9304 | 3.2483 | 0.43 |
| Ridge | rmse | 54.9362 | 3.0363 | 0.05 |

## Random Forest vs best gradient booster (Wilcoxon signed-rank on paired folds)

| dataset | metric | rf_mean | best_booster | booster_mean | winner | wilcoxon_p | significant_5pct |
|---|---|---|---|---|---|---|---|
| breast_cancer | roc_auc | 0.9898 | CatBoost | 0.9943 | CatBoost | 0.0231 | True |
| diabetes | rmse | 57.8605 | CatBoost | 58.7170 | Random Forest | 0.2769 | False |
| lk_colombo_dengue | rmse | 62.3788 | CatBoost | 62.6423 | Random Forest | 0.8040 | False |
| lk_tea_auction_catalogue | roc_auc | 0.9614 | CatBoost | 0.9736 | CatBoost | 0.0001 | True |
| lk_tea_auction_price | rmse | 104.7598 | LightGBM | 97.6938 | LightGBM | 0.0001 | True |
| wine | roc_auc | 0.9981 | CatBoost | 0.9981 | CatBoost | 0.9165 | False |

## Ablation: what preprocessing and leakage control contribute (tea catalogue, RF, test macro-F1)

| setup | test_f1_macro |
|---|---|
| TabulaML automatic pipeline (leaky 'category' excluded) | 0.8005 |
| Same pipeline but 'category' kept (inflated: it encodes the catalogue) | 1.0000 |
| Naive: drop rows with any NaN (436 lost), one-hot every column as text | 0.7678 |
