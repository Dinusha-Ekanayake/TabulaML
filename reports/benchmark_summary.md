# Benchmark: Random Forest vs gradient boosting

Same engine as the app: dev-only automatic configuration, 5-fold cross-validation on the training rows with each dataset's validation design, then one score on the held-out test rows. RMSE: lower is better; ROC-AUC, F1, R2: higher is better.

## lk_colombo_dengue (real, public (Sri Lanka), time_ordered validation)

| model | metric | cv_mean | cv_std | test | fit_s |
|---|---|---|---|---|---|
| Random Forest | rmse | 64.3231 | 19.4407 | 65.1748 | 0.3866 |
| XGBoost | rmse | 71.9845 | 21.7386 | 68.0546 | 0.7356 |
| LightGBM | rmse | 70.0845 | 16.3904 | 86.7866 | 0.1699 |
| CatBoost | rmse | 68.1472 | 20.0989 | 69.3478 | 1.0589 |
| Gradient Boosting | rmse | 67.0541 | 21.3752 | 67.8799 | 0.1059 |
| Ridge Regression | rmse | 60.7192 | 16.7921 | 74.2196 | 0.0174 |

## lk_tea_auction_catalogue (real, public (Sri Lanka), grouped validation)

| model | metric | cv_mean | cv_std | test | fit_s |
|---|---|---|---|---|---|
| Random Forest | f1_macro | 0.8549 | 0.0042 | 0.8559 | 1.3150 |
| XGBoost | f1_macro | 0.8572 | 0.0028 | 0.8518 | 3.0407 |
| LightGBM | f1_macro | 0.8533 | 0.0080 | 0.8447 | 4.4204 |
| CatBoost | f1_macro | 0.8529 | 0.0030 | 0.8513 | 5.9705 |
| Gradient Boosting | f1_macro | 0.8499 | 0.0051 | 0.8428 | 7.8389 |
| Logistic Regression | f1_macro | 0.8410 | 0.0084 | 0.8433 | 0.5607 |

## lk_tea_auction_price (real, public (Sri Lanka), grouped validation)

| model | metric | cv_mean | cv_std | test | fit_s |
|---|---|---|---|---|---|
| Random Forest | rmse | 125.8608 | 20.4382 | 102.2526 | 4.6188 |
| XGBoost | rmse | 130.6290 | 21.3340 | 115.9429 | 1.3191 |
| LightGBM | rmse | 123.1606 | 17.7670 | 106.4342 | 1.1606 |
| CatBoost | rmse | 129.5920 | 19.2447 | 112.8540 | 2.6045 |
| Gradient Boosting | rmse | 156.7411 | 13.4703 | 167.5980 | 1.7776 |
| Ridge Regression | rmse | 234.2600 | 5.9460 | 255.9064 | 0.2486 |

## breast_cancer (public benchmark (sklearn), random validation)

| model | metric | cv_mean | cv_std | test | fit_s |
|---|---|---|---|---|---|
| Random Forest | roc_auc | 0.9896 | 0.0131 | 0.9937 | 0.4499 |
| XGBoost | roc_auc | 0.9939 | 0.0081 | 0.9950 | 0.4871 |
| LightGBM | roc_auc | 0.9938 | 0.0070 | 0.9894 | 0.3719 |
| CatBoost | roc_auc | 0.9941 | 0.0075 | 0.9967 | 2.0496 |
| Gradient Boosting | roc_auc | 0.9919 | 0.0095 | 0.9907 | 0.5124 |
| Logistic Regression | roc_auc | 0.9955 | 0.0061 | 0.9954 | 0.0427 |

## wine (public benchmark (sklearn), random validation)

| model | metric | cv_mean | cv_std | test | fit_s |
|---|---|---|---|---|---|
| Random Forest | f1_macro | 0.9794 | 0.0304 | 1.0000 | 0.4205 |
| XGBoost | f1_macro | 0.9794 | 0.0304 | 0.9487 | 0.5599 |
| LightGBM | f1_macro | 0.9791 | 0.0309 | 0.9487 | 0.2579 |
| CatBoost | f1_macro | 0.9657 | 0.0419 | 0.9230 | 1.2856 |
| Gradient Boosting | f1_macro | 0.9410 | 0.0354 | 0.9743 | 0.2597 |
| Logistic Regression | f1_macro | 0.9658 | 0.0249 | 0.9457 | 0.0147 |

## diabetes (public benchmark (sklearn), random validation)

| model | metric | cv_mean | cv_std | test | fit_s |
|---|---|---|---|---|---|
| Random Forest | rmse | 59.9738 | 4.9492 | 54.7580 | 0.3911 |
| XGBoost | rmse | 61.8209 | 3.1739 | 56.9478 | 0.6083 |
| LightGBM | rmse | 62.3258 | 3.3387 | 57.7478 | 0.2401 |
| CatBoost | rmse | 59.5959 | 3.3256 | 53.6431 | 1.0319 |
| Gradient Boosting | rmse | 62.7962 | 3.1259 | 53.8608 | 0.0892 |
| Ridge Regression | rmse | 55.8926 | 4.6618 | 53.7775 | 0.0086 |

## Random Forest vs each booster (cross-validation folds)

'RF better in k of n folds' counts the folds where Random Forest scored better. Verdict: RF/booster better when it wins at least 80% of folds, otherwise 'about the same'.

| dataset | metric | booster | rf_cv_mean | booster_cv_mean | folds_rf_won | folds_booster_won | folds_tied | verdict |
|---|---|---|---|---|---|---|---|---|
| lk_colombo_dengue | rmse | xgb | 64.3231 | 71.9845 | 5 | 0 | 0 | RF better |
| lk_colombo_dengue | rmse | lgbm | 64.3231 | 70.0845 | 4 | 1 | 0 | RF better |
| lk_colombo_dengue | rmse | catboost | 64.3231 | 68.1472 | 3 | 2 | 0 | about the same |
| lk_tea_auction_catalogue | f1_macro | xgb | 0.8549 | 0.8572 | 1 | 4 | 0 | booster better |
| lk_tea_auction_catalogue | f1_macro | lgbm | 0.8549 | 0.8533 | 2 | 3 | 0 | about the same |
| lk_tea_auction_catalogue | f1_macro | catboost | 0.8549 | 0.8529 | 4 | 1 | 0 | RF better |
| lk_tea_auction_price | rmse | xgb | 125.8608 | 130.6290 | 3 | 2 | 0 | about the same |
| lk_tea_auction_price | rmse | lgbm | 125.8608 | 123.1606 | 1 | 4 | 0 | booster better |
| lk_tea_auction_price | rmse | catboost | 125.8608 | 129.5920 | 5 | 0 | 0 | RF better |
| breast_cancer | roc_auc | xgb | 0.9896 | 0.9939 | 0 | 4 | 1 | booster better |
| breast_cancer | roc_auc | lgbm | 0.9896 | 0.9938 | 1 | 3 | 1 | about the same |
| breast_cancer | roc_auc | catboost | 0.9896 | 0.9941 | 0 | 3 | 2 | about the same |
| wine | f1_macro | xgb | 0.9794 | 0.9794 | 1 | 1 | 3 | about the same |
| wine | f1_macro | lgbm | 0.9794 | 0.9791 | 1 | 1 | 3 | about the same |
| wine | f1_macro | catboost | 0.9794 | 0.9657 | 2 | 1 | 2 | about the same |
| diabetes | rmse | xgb | 59.9738 | 61.8209 | 4 | 1 | 0 | RF better |
| diabetes | rmse | lgbm | 59.9738 | 62.3258 | 4 | 1 | 0 | RF better |
| diabetes | rmse | catboost | 59.9738 | 59.5959 | 2 | 3 | 0 | about the same |

## Ablation on the tea catalogue task (Random Forest, macro-F1)

| setup | cv_f1_macro | test_f1_macro |
|---|---|---|
| TabulaML: grouped validation, leaky 'category' excluded | 0.8549 | 0.8559 |
| Same, but 'category' kept (it encodes the answer) | 1.0000 | 1.0000 |
| Random split that ignores the weekly-sale groups | 0.8085 | 0.8206 |
