# register_model_mlflow_toy.py
# Works perfectly on Windows / macOS
# Fully suppresses all MLflow warnings and log noise for a clean console

import warnings
import logging
import os

# Suppress Python warnings
warnings.filterwarnings("ignore")

# Suppress all MLflow / SQLAlchemy / urllib logs globally
logging.disable(logging.CRITICAL)

# Optional: disable MLflow env verbose logs
os.environ["MLFLOW_TRACKING_VERBOSE"] = "0"

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Connect to your running MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create or use experiment
mlflow.set_experiment("ToyModelRegistryExperiment")

# Load dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run() as run:
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Log model in MLflow
    mlflow.sklearn.log_model(model, "rf_model")

    # Register model in the Model Registry
    try:
        result = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/rf_model",
            name="MyModelRegistry"
        )
        print(f"Model registered successfully! Version: {result.version}")
    except Exception as e:
        print("Failed to register model.")
        print(f"Error: {e}")

    print(f"🏃 View run at: http://127.0.0.1:5000/#/experiments/2/runs/{run.info.run_id}")
    print(f"View experiment at: http://127.0.0.1:5000/#/experiments/2")