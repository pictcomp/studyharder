# track_experiments_wandb.py
# Works perfectly on Windows and VS Code with W&B and sklearn

import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize W&B project
wandb.init(project="WineQualityTracking", name="RandomForest_Experiment")

# Load dataset
data = pd.read_csv(r"FILEPATH.csv")

# Prepare data
X = data.drop("quality", axis=1)
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set hyperparameters
config = {
    "n_estimators": 100,
    "max_depth": 5,
    "test_size": 0.2,
    "random_state": 42
}
wandb.config.update(config)

# Train model
model = RandomForestClassifier(
    n_estimators=config["n_estimators"],
    max_depth=config["max_depth"],
    random_state=config["random_state"]
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
wandb.log({"accuracy": accuracy})

print(f"Model trained and logged successfully! Accuracy: {accuracy:.4f}")

# Save model (optional)
import joblib
joblib.dump(model, "rf_model.joblib")
wandb.save("rf_model.joblib")

wandb.finish()
