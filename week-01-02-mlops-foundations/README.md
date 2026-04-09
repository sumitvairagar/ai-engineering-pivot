# Week 1–2: MLOps Foundations + MLflow

> **Theme:** Understand the ML lifecycle from YOUR lens — a senior engineer who already knows pipelines, CI/CD, and distributed systems.
> **Time commitment:** ~1.5–2 hrs/day, or batch it on weekends
> **Status:** ⬜ Not Started

---

## 🎬 Demoable Deliverable

**What to show:** MLflow dashboard with your OrderFillPredictor experiments — multiple runs compared side-by-side, metrics visualized, best model promoted to Staging.

**Demo format:**
- Screenshot or screen recording of MLflow UI showing experiment comparison
- Evidently drift report (HTML) showing detected data drift
- Push to GitHub with a clear README explaining what each experiment tested

**Where to share:** GitHub repo + LinkedIn post teaser ("I tracked 6 ML experiments on brokerage data using MLflow — here's what I learned about model versioning")

---

## 🎯 What You're Learning (and Why)

You already build event pipelines in Kafka. MLOps is the same idea applied to ML models:

| Concept You Know | MLOps Equivalent |
|------------------|-----------------|
| Kafka topic = data stream | Feature pipeline = data in/out of model |
| Spring Boot service | Model serving API |
| Git versioning | Model versioning (MLflow) |
| CI/CD pipeline | ML pipeline (train → evaluate → deploy) |
| Observability/logs | Model monitoring + drift detection |

**MLflow** is your entry point. It's an open-source tool that tracks experiments, versions models, and manages deployment. Think of it as "Git + logging + artifact storage" for ML models.

---

## ✅ Week 1 Checklist

### 🔧 Setup (Day 1)
- [ ] Install Python 3.10+ and set up a virtual environment
  ```bash
  python -m venv mlops-env
  source mlops-env/bin/activate
  pip install mlflow scikit-learn pandas numpy matplotlib
  ```
- [ ] Run MLflow UI locally — open http://localhost:5000
  ```bash
  mlflow ui
  ```
- [ ] Read: [MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html) — first 2 sections only (~30 min)
- [ ] **What to absorb:** How `mlflow.log_param()`, `mlflow.log_metric()`, and `mlflow.log_model()` work. That's 80% of MLflow you'll use daily.

### 🧪 First Experiment (Day 2–3)
- [ ] Create `experiments/first_run.py` with this code and run it:

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_experiment("iris-classifier")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try different values — run this script 3 times with 50, 100, 200
n_estimators = 100

with mlflow.start_run():
    mlflow.log_param("n_estimators", n_estimators)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    print(f"Accuracy: {acc:.4f}")
```

- [ ] Open MLflow UI, go to the "iris-classifier" experiment
- [ ] See all 3 runs listed with different params and metrics
- [ ] Click "Compare" on 2 runs — observe the diff view
- [ ] **Key insight to write in notes.md:** This is model versioning. Same as comparing git commits, but for trained models.

### 📖 Understand the ML Lifecycle (Day 4–5)
- [ ] Read: [Made With ML — MLOps Motivation](https://madewithml.com/courses/mlops/motivation/) (~20 min)
- [ ] Read: [Made With ML — Design section](https://madewithml.com/courses/mlops/design/) (~30 min)
- [ ] Watch (optional, 15 min): Search YouTube for "MLOps explained in 10 minutes" — Weights & Biases or Google Cloud versions are good
- [ ] **What to absorb:** The 4 stages — Data → Training → Evaluation → Deployment. Know what goes wrong at each stage in production.

### 💭 Concept: Feature Pipelines (Day 6–7)
- [ ] Read: [What is a Feature Store](https://www.featurestore.org/what-is-a-feature-store) (~15 min)
- [ ] Read: [Feature Pipelines vs Data Pipelines](https://www.tecton.ai/blog/what-is-a-feature-pipeline/) (~10 min)
- [ ] **Think exercise — do this in notes.md:**
  In the Manafsoft OMS context, define 5 "features" a model might need:
  - Example: `order_volume_last_1hr`, `broker_fill_rate_7d`, `market_volatility_score`
  - Write your own 5. This is the domain-specific thinking that makes you valuable.

---

## ✅ Week 2 Checklist

### 📦 MLflow Model Registry (Day 1–2)
- [ ] Read: [MLflow Model Registry docs](https://mlflow.org/docs/latest/model-registry.html) — first 3 sections
- [ ] Register your best iris run in the registry:
```python
# Get the run_id from the MLflow UI, then:
model_uri = "runs:/<your_run_id>/model"
mlflow.register_model(model_uri, "IrisClassifier")
```
- [ ] In the MLflow UI, go to Models tab → transition to Staging → then Production
- [ ] **Key insight:** Staging = your test environment. Production = live. Exact same mental model as your current deployment pipelines.

### 🏦 Domain Experiment: Order Fill Rate Predictor (Day 3–4)

This is your first piece of portfolio work. Synthetic brokerage data, mimicking Manafsoft context.

- [ ] Create `experiments/order_fill_predictor.py`:

```python
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
n = 2000

data = pd.DataFrame({
    'order_size': np.random.randint(100, 10000, n),
    'broker_tier': np.random.choice(['A', 'B', 'C'], n),
    'market_volatility': np.random.uniform(0.1, 2.0, n),
    'time_of_day_hour': np.random.randint(9, 16, n),
    'order_type': np.random.choice(['market', 'limit', 'stop'], n),
    'days_since_last_trade': np.random.randint(0, 30, n),
})

# Target: will this order be filled?
data['filled'] = (
    (data['order_size'] < 5000) & 
    (data['market_volatility'] < 1.5) &
    (data['order_type'] == 'market')
).astype(int)
# Add noise
noise_idx = np.random.choice(n, size=int(n*0.15), replace=False)
data.loc[noise_idx, 'filled'] = 1 - data.loc[noise_idx, 'filled']

le_broker = LabelEncoder()
le_type = LabelEncoder()
data['broker_tier_enc'] = le_broker.fit_transform(data['broker_tier'])
data['order_type_enc'] = le_type.fit_transform(data['order_type'])

features = ['order_size', 'broker_tier_enc', 'market_volatility',
            'time_of_day_hour', 'order_type_enc', 'days_since_last_trade']
X = data[features]
y = data['filled']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
```

- [ ] Run it — you'll see 6 runs in MLflow UI
- [ ] Find the best model by F1 score (better metric for imbalanced classes than accuracy)
- [ ] Register the best as `OrderFillPredictor` and promote to Staging
- [ ] **Portfolio note:** Add a comment in the file explaining why you used GradientBoosting over RandomForest for this use case. This shows engineering judgment.

### 🔍 Model Monitoring + Drift (Day 5)
- [ ] Install Evidently:
  ```bash
  pip install evidently
  ```
- [ ] Create `experiments/drift_demo.py`:

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import pandas as pd
import numpy as np

np.random.seed(42)

# Reference = last month's data distribution
reference = pd.DataFrame({
    'order_size': np.random.normal(5000, 1000, 500),
    'market_volatility': np.random.normal(1.0, 0.3, 500),
    'days_since_last_trade': np.random.exponential(5, 500),
})

# Current = this week — market conditions changed!
current = pd.DataFrame({
    'order_size': np.random.normal(7500, 2000, 500),  # bigger orders
    'market_volatility': np.random.normal(1.8, 0.5, 500),  # more volatile!
    'days_since_last_trade': np.random.exponential(5, 500),
})

report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
report.run(reference_data=reference, current_data=current)
report.save_html("drift_report.html")
print("Open drift_report.html in your browser")
```

- [ ] Open `drift_report.html` — observe which features drifted
- [ ] **Key insight to write in notes.md:** If `market_volatility` drifts significantly, your `OrderFillPredictor` predictions become unreliable. This is why you need monitoring, not just deployment.

### 📝 Week 2 Wrap-up (Day 6–7)
- [ ] Write `notes.md` answers:
  1. How would you integrate MLflow into a Java/Spring Boot CI/CD pipeline? (Hint: MLflow has a REST API)
  2. What would you monitor in production for the OrderFillPredictor?
  3. What's the difference between model accuracy dropping vs data drift? Which do you detect first?
- [ ] Commit all code to GitHub
- [ ] Update main README dashboard → Week 1–2 ✅ Done

---

## 📚 Resource Reference

| Resource | Type | Est. Time | Priority |
|----------|------|-----------|----------|
| [MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html) | Docs | 30 min | 🔴 Must |
| [Made With ML — MLOps](https://madewithml.com/courses/mlops/motivation/) | Guide | 1 hr | 🔴 Must |
| [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html) | Docs | 30 min | 🔴 Must |
| [Evidently AI Blog — Drift](https://evidentlyai.com/blog/ml-monitoring-data-drift-detection) | Blog | 20 min | 🟡 Recommended |
| [Feature Store Explained](https://www.featurestore.org/what-is-a-feature-store) | Concept | 15 min | 🟡 Recommended |
| [Tecton — Feature Pipelines](https://www.tecton.ai/blog/what-is-a-feature-pipeline/) | Blog | 10 min | 🟢 Optional |

---

## 🏁 End-of-Phase Self-Assessment

Before moving to Week 3–4, confirm you can say YES to all of these:

- [ ] I can track ML experiments with MLflow and compare runs
- [ ] I understand how model versioning maps to my existing CI/CD knowledge
- [ ] I've run a drift report and know why it matters in production
- [ ] I've built an experiment on domain-relevant (financial) data
- [ ] I can explain MLOps to a non-ML engineer in 2 minutes

**Next:** [Week 3–4 — Model Serving with FastAPI + Docker →](../week-03-04-model-serving/README.md)
