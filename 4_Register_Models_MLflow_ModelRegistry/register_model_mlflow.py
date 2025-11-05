# register_model_mlflow.py
# Works perfectly on Windows and VS Code
# Trains, logs, and registers a RandomForest model on the Wine Quality dataset using MLflow Model Registry.
# Suppresses MLflow deprecation and signature warnings for cleaner output

import warnings

# Suppress MLflow and sklearn warnings
warnings.filterwarnings("ignore", message=".*artifact_path.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Model logged without a signature.*")
warnings.filterwarnings("ignore", message=".*mlflow.tracking._model_registry.fluent.*")

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Connect to your MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set experiment
mlflow.set_experiment("WineModelRegistryExperiment")

# Load dataset
data = pd.read_csv(r"FILEPATH.csv")

# Prepare features and labels
X = data.drop("quality", axis=1)
y = data["quality"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
n_estimators = 100
max_depth = 6
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model trained successfully! Accuracy: {acc:.4f}")

# Log and register in MLflow
with mlflow.start_run() as run:
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "rf_model")

    print(f"🔁 Logging model to MLflow... Run ID: {run.info.run_id}")

    try:
        result = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/rf_model",
            name="WineQualityModel"
        )
        print(f"Model registered successfully! Name: {result.name}, Version: {result.version}")
    except Exception as e:
        print("Model registration failed.")
        print(f"Error: {e}")