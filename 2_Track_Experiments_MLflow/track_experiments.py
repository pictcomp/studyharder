# track_experiments.py
# Works perfectly on Windows with MLflow and sklearn

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set experiment name
mlflow.set_experiment("WineQualityExperiment")

# Load dataset (update path if needed)
data = pd.read_csv(r"winequality-red.csv")

# Prepare features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameters
n_estimators = 100
max_depth = 5

# Start MLflow experiment
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log parameters, metrics, and model
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Model logged successfully! Accuracy: {acc:.4f}")
