# Forecasting benchmark: Colombo weekly dengue cases

Time-ordered 5-fold CV on the training weeks; test = latest 20 % of weeks. Features: target lags h..h+3, seasonal lag 52, 4-week rolling mean, calendar parts, weather lagged by h. `vs_naive` = RMSE improvement over the naive baseline on the test weeks (positive = better).

| horizon | model | cv_rmse | test_rmse | test_mae | vs_naive |
|---|---|---|---|---|---|
| 1 | Naive forecast (last known value) | 62.6 ± 23.6 | 62.1 | 46.5 | +0.0% |
| 1 | Seasonal naive | 90.1 ± 45.1 | 98.7 | 79.2 | -58.9% |
| 1 | Ridge Regression | 72.9 ± 14.5 | 83.9 | 64.4 | -35.1% |
| 1 | Random Forest | 63.6 ± 20.9 | 67.3 | 48.8 | -8.3% |
| 1 | XGBoost | 68.6 ± 23.6 | 71.5 | 53.9 | -15.1% |
| 1 | Hybrid RF + ARIMA | 61.6 ± 20.0 | 60.4 | 46.2 | +2.8% |
| 4 | Naive forecast (last known value) | 85.8 ± 19.6 | 79.0 | 66.5 | +0.0% |
| 4 | Seasonal naive | 92.5 ± 42.9 | 99.1 | 79.4 | -25.5% |
| 4 | Ridge Regression | 94.4 ± 38.3 | 92.9 | 69.3 | -17.6% |
| 4 | Random Forest | 76.2 ± 18.3 | 83.7 | 62.4 | -6.0% |
| 4 | XGBoost | 74.1 ± 17.6 | 80.5 | 59.0 | -1.9% |
| 4 | Hybrid RF + ARIMA | 74.7 ± 21.1 | 72.8 | 62.6 | +7.8% |
