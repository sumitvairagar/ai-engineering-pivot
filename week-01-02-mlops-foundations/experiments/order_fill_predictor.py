import pandas as pd
import numpy as np

np.random.seed(42) #make results reproducible
n = 2000

data = pd.DataFrame({
    'order_size': np.random.randint(100, 10000, n),
    'broker_tier': np.random.choice(['A', 'B', 'C'], n),
    'market_volatility': np.random.uniform(0.1, 2.0, n),
    'time_of_day_hour': np.random.randint(9, 16, n),
    'order_type': np.random.choice(['market', 'limit', 'stop'], n),
    'days_since_last_trade': np.random.randint(0, 30, n),
})

print(data.head())
print(f"\nShape: {data.shape}")
print(f"\nColumn types:\n{data.dtypes}")


# Target: will this order be filled?
# Rules that mimic real market behavior:
# - Small orders + low volatility + market orders = likely filled
data['filled'] = (
    (data['order_size'] < 5000) & 
    (data['market_volatility'] < 1.5) &
    (data['order_type'] == 'market')
).astype(int)

# Add 15% noise — real data is never perfectly clean
noise_idx = np.random.choice(n, size=int(n * 0.15), replace=False)
data.loc[noise_idx, 'filled'] = 1 - data.loc[noise_idx, 'filled']

print(f"\nFill rate: {data['filled'].mean():.1%}")
print(f"\nFilled counts:\n{data['filled'].value_counts()}")


from sklearn.preprocessing import LabelEncoder

le_broker = LabelEncoder()
le_type = LabelEncoder()
data['broker_tier_enc'] = le_broker.fit_transform(data['broker_tier'])  # A=0, B=1, C=2
data['order_type_enc'] = le_type.fit_transform(data['order_type'])      # limit=0, market=1, stop=2

print(f"\nBroker mapping: {dict(zip(le_broker.classes_, le_broker.transform(le_broker.classes_)))}")
print(f"Order type mapping: {dict(zip(le_type.classes_, le_type.transform(le_type.classes_)))}")


from sklearn.model_selection import train_test_split

features = ['order_size', 'broker_tier_enc', 'market_volatility',
            'time_of_day_hour', 'order_type_enc', 'days_since_last_trade']
X = data[features]   # inputs
y = data['filled']    # what we're predicting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape[0]} orders")
print(f"Test set: {X_test.shape[0]} orders")


import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Why GradientBoosting over RandomForest:
# GradientBoosting builds trees sequentially — each tree corrects the previous one's mistakes.
# For tabular financial data with mixed feature types and class imbalance (22% fill rate),
# it consistently outperforms RandomForest. In production, XGBoost/LightGBM (faster variants)
# are the industry standard for structured data like order books and trade records.

mlflow.set_experiment("order-fill-predictor")

for lr in [0.05, 0.1, 0.2]:
    for depth in [3, 5]:
        with mlflow.start_run(run_name=f"lr={lr}_depth={depth}"):
            mlflow.log_params({"learning_rate": lr, "max_depth": depth, "domain": "brokerage-oms"})
            
            model = GradientBoostingClassifier(learning_rate=lr, max_depth=depth, random_state=42)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            
            mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
            mlflow.sklearn.log_model(model, "model")
            print(f"lr={lr}, depth={depth}: acc={acc:.3f}, f1={f1:.3f}")
