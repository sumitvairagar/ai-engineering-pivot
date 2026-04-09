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

## Drift Detection

- **Data drift** = your production data looks different from training data. Model still runs, no errors, but predictions become unreliable.
- Evidently compares reference (training-time) vs current (production) data distributions.
- In our demo: `order_size` and `market_volatility` drifted, `days_since_last_trade` didn't.
- This is your early warning to retrain — before accuracy drops and someone notices.

## Week 2 Wrap-up Questions

**1. How to integrate MLflow into a Java/Spring Boot CI/CD pipeline?**
- MLflow has a REST API. Training pipeline (Python) runs in CI, logs experiments.
- Spring Boot app loads the production model via REST: "give me OrderFillPredictor@production."
- On model promotion, app picks up new version on restart or scheduled refresh.
- Same pattern as reading config from Spring Cloud Config Server.

**2. What to monitor in production for OrderFillPredictor?**
- Data drift — are inputs changing? (Evidently)
- Prediction distribution — is the model suddenly predicting differently than usual?
- Actual accuracy — compare predictions to ground truth when it arrives (delayed feedback).

**3. Model accuracy drop vs data drift — which comes first?**
- Data drift = inputs changed. Accuracy drop = outputs are wrong.
- You detect drift FIRST — it's visible immediately in the data.
- Accuracy drop is only measurable after you get actual outcomes (hours/days later).
- Drift is the leading indicator, accuracy is the lagging indicator.
