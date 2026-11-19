# Benchmark: Random Forest vs gradient boosting

Repeated cross-validation (5 folds x 3 repeats, seed 42); identical folds for every model; TabulaML Automatic-mode preprocessing for every dataset. Error metrics (RMSE) are lower-is-better; ROC-AUC, F1 and R2 are higher-is-better.

## lk_colombo_dengue (real, public (Sri Lanka))

| Model | Metric | Mean | Std | Fit time (s) |
|---|---|---|---|---|
| Random Forest | rmse | 62.3788 | 7.3387 | 0.52 |
| XGBoost | rmse | 64.4065 | 7.8262 | 0.85 |
| LightGBM | rmse | 67.4801 | 6.8945 | 0.41 |
| CatBoost | rmse | 62.6423 | 6.9696 | 1.12 |
| Gradient Boosting | rmse | 64.9620 | 6.4081 | 0.18 |
| Ridge | rmse | 66.8489 | 13.4787 | 0.04 |

## lk_synthetic_patient_los (synthetic (SL context))

| Model | Metric | Mean | Std | Fit time (s) |
|---|---|---|---|---|
| Random Forest | rmse | 3.0646 | 0.1850 | 1.39 |
| XGBoost | rmse | 2.9929 | 0.1775 | 1.22 |
| LightGBM | rmse | 3.0174 | 0.1694 | 2.08 |
| CatBoost | rmse | 2.9194 | 0.1910 | 2.28 |
| Gradient Boosting | rmse | 2.9291 | 0.1896 | 1.06 |
| Ridge | rmse | 2.8493 | 0.1827 | 0.21 |

## lk_synthetic_telecom_churn (synthetic (SL context))

| Model | Metric | Mean | Std | Fit time (s) |
|---|---|---|---|---|
| Random Forest | roc_auc | 0.6653 | 0.0172 | 1.18 |
| XGBoost | roc_auc | 0.6600 | 0.0177 | 1.31 |
| LightGBM | roc_auc | 0.6583 | 0.0203 | 2.02 |
| CatBoost | roc_auc | 0.6750 | 0.0172 | 2.37 |
| Gradient Boosting | roc_auc | 0.6860 | 0.0190 | 1.82 |
| Logistic Regression | roc_auc | 0.6945 | 0.0160 | 0.39 |

## breast_cancer (public benchmark (sklearn))

| Model | Metric | Mean | Std | Fit time (s) |
|---|---|---|---|---|
| Random Forest | roc_auc | 0.9898 | 0.0092 | 0.61 |
| XGBoost | roc_auc | 0.9939 | 0.0049 | 0.51 |
| LightGBM | roc_auc | 0.9936 | 0.0048 | 0.78 |
| CatBoost | roc_auc | 0.9943 | 0.0044 | 2.23 |
| Gradient Boosting | roc_auc | 0.9924 | 0.0052 | 0.72 |
| Logistic Regression | roc_auc | 0.9952 | 0.0045 | 0.07 |

## wine (public benchmark (sklearn))

| Model | Metric | Mean | Std | Fit time (s) |
|---|---|---|---|---|
| Random Forest | roc_auc | 0.9981 | 0.0035 | 0.58 |
| XGBoost | roc_auc | 0.9962 | 0.0047 | 0.75 |
| LightGBM | roc_auc | 0.9968 | 0.0047 | 0.60 |
| CatBoost | roc_auc | 0.9981 | 0.0027 | 1.13 |
| Gradient Boosting | roc_auc | 0.9956 | 0.0055 | 0.41 |
| Logistic Regression | roc_auc | 0.9973 | 0.0032 | 0.02 |

## diabetes (public benchmark (sklearn))

| Model | Metric | Mean | Std | Fit time (s) |
|---|---|---|---|---|
| Random Forest | rmse | 57.8605 | 3.4269 | 0.51 |
| XGBoost | rmse | 60.1238 | 3.8883 | 0.69 |
| LightGBM | rmse | 62.2977 | 3.5008 | 0.53 |
| CatBoost | rmse | 58.7170 | 3.9857 | 0.92 |
| Gradient Boosting | rmse | 58.9304 | 3.2483 | 0.15 |
| Ridge | rmse | 54.9362 | 3.0363 | 0.01 |

## Random Forest vs best gradient booster (Wilcoxon signed-rank on paired folds)

| dataset | metric | rf_mean | best_booster | booster_mean | winner | wilcoxon_p | significant_5pct |
|---|---|---|---|---|---|---|---|
| breast_cancer | roc_auc | 0.9898 | CatBoost | 0.9943 | CatBoost | 0.0231 | True |
| diabetes | rmse | 57.8605 | CatBoost | 58.7170 | Random Forest | 0.2769 | False |
| lk_colombo_dengue | rmse | 62.3788 | CatBoost | 62.6423 | Random Forest | 0.8040 | False |
| lk_synthetic_patient_los | rmse | 3.0646 | CatBoost | 2.9194 | CatBoost | 0.0001 | True |
| lk_synthetic_telecom_churn | roc_auc | 0.6653 | CatBoost | 0.6750 | CatBoost | 0.0026 | True |
| wine | roc_auc | 0.9981 | CatBoost | 0.9981 | CatBoost | 0.9165 | False |

## Ablation: what preprocessing and leakage control contribute (synthetic churn, RF, test ROC-AUC)

| setup | test_roc_auc |
|---|---|
| TabulaML automatic pipeline (leak excluded) | 0.6446 |
| Same pipeline but leakage column kept (inflated, invalid) | 1.0000 |
| Naive: drop NaN rows (857 lost), one-hot everything as text | 0.6271 |
