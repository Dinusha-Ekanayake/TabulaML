# Forecasting benchmark: Colombo weekly dengue cases

Time-ordered 5-fold CV on the training weeks; test = latest 20 % of weeks. Features: target lags h..h+3, seasonal lag 52, 4-week rolling mean, calendar parts, weather lagged by h. `vs_naive` = RMSE improvement over the naive baseline on the test weeks (positive = better).

| horizon | model | cv_rmse | test_rmse | test_mae | vs_naive |
|---|---|---|---|---|---|
| 1 | Naive forecast (last known value) | 62.6 ± 23.6 | 62.1 | 46.5 | +0.0% |
| 1 | Seasonal naive | 90.1 ± 45.1 | 98.7 | 79.2 | -58.9% |
| 1 | Ridge Regression | 75.9 ± 16.6 | 84.2 | 64.7 | -35.5% |
| 1 | Random Forest | 63.6 ± 20.9 | 67.3 | 48.8 | -8.3% |
| 1 | XGBoost | 68.6 ± 23.6 | 71.5 | 53.9 | -15.1% |
| 1 | Hybrid RF + ARIMA | 61.6 ± 20.0 | 60.4 | 46.2 | +2.8% |
| 4 | Naive forecast (last known value) | 85.7 ± 19.8 | 79.0 | 66.5 | +0.0% |
| 4 | Seasonal naive | 91.3 ± 45.3 | 99.1 | 79.4 | -25.5% |
| 4 | Ridge Regression | 105.7 ± 27.1 | 91.0 | 68.3 | -15.3% |
| 4 | Random Forest | 77.1 ± 20.2 | 84.4 | 63.0 | -6.9% |
| 4 | XGBoost | 74.6 ± 19.5 | 83.2 | 61.3 | -5.3% |
| 4 | Hybrid RF + ARIMA | 74.9 ± 18.8 | 73.6 | 64.0 | +6.8% |
