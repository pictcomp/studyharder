# register_model_mlflow_toy.py
# ✅ Works perfectly on Windows / macOS
# Trains, logs, and registers a RandomForestRegressor in MLflow Model Registry
# Clean, quiet console output (no warnings or verbose MLflow logs)

import warnings
import logging
import os

# Suppress warnings and logs for a clean console
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["MLFLOW_TRACKING_VERBOSE"] = "0"

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create or use experiment
mlflow.set_experiment("ToyModelRegistryExperiment")

# Load dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters
n_estimators = 100
max_depth = 6

# Start MLflow run
with mlflow.start_run() as run:
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2_Score", r2)

    # Log model
    mlflow.sklearn.log_model(model, "rf_model")

    # Register model in the Model Registry
    try:
        result = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/rf_model",
            name="MyModelRegistry"
        )
        print(f"✅ Model registered successfully! Version: {result.version}")
    except Exception as e:
        print("❌ Failed to register model.")
        print(f"Error: {e}")

    print("\n📊 Metrics logged to MLflow:")
    print(f"   MAE   : {mae:.4f}")
    print(f"   RMSE  : {rmse:.4f}")
    print(f"   R²    : {r2:.4f}")
    print("\n🌐 View in MLflow UI:")
    print(f"   Run: http://127.0.0.1:5000/#/experiments/2/runs/{run.info.run_id}")
    print(f"   Experiment: http://127.0.0.1:5000/#/experiments/2")
