# Week 1-2: MLOps Foundations — Notes

## Key Insights

- **MLflow experiment tracking = model versioning.** Same as comparing git commits, but for trained models. Each run captures params (config), metrics (results), and artifacts (the model itself).
- Core API is just 3 calls: `log_param()`, `log_metric()`, `log_model()` — wrapped in a `start_run()` context manager.

## Feature Pipeline Concepts

- **Feature** = an input column to the model. Just a number/value the model uses to predict.
- **Feature pipeline** = code that computes features from raw data (same as Kafka Streams transforms, but output feeds an ML model).
- **Feature store** = a cache/DB for pre-computed features so the model can look them up fast.

## 5 Brokerage Features for Order Fill Prediction

1. `time_of_day_hour` — market open has more liquidity than close
2. `asking_price_vs_market` — spread between limit price and current market price
3. `available_quantity` — how much is on the order book at that price level
4. `order_size_to_available_ratio` — order size / available quantity (derived feature)
5. `market_volatility_last_15min` — recent price movement intensity
