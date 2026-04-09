import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

mlflow.set_experiment("iris-classifier")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pass n_estimators as CLI arg, default 100
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100

with mlflow.start_run():
    mlflow.log_param("n_estimators", n_estimators)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    print(f"n_estimators={n_estimators}, accuracy={acc:.4f}")
