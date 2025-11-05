# track_experiments_wandb_toy.py
# Self-contained example using Iris dataset and W&B

import wandb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize W&B project
wandb.init(project="IrisTracking", name="Iris_RF_Experiment")

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and log hyperparameters
config = {"n_estimators": 50, "max_depth": 4, "test_size": 0.2}
wandb.config.update(config)

# Train model
model = RandomForestClassifier(n_estimators=config["n_estimators"], max_depth=config["max_depth"], random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
wandb.log({"accuracy": accuracy})

print(f"Iris model trained! Accuracy: {accuracy:.4f}")

wandb.finish()
