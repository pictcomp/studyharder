# automate_prefect_toy.py
# Automates preprocessing and training using Prefect with Iris dataset
# Suppresses Prefect async SQLAlchemy shutdown warnings

import warnings

# Hide the harmless async shutdown warning from Prefect
warnings.filterwarnings("ignore", message=".*sqlalchemy.*CancelledError.*")

from prefect import flow, task
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
@task
def load_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    return X, y


# Normalize features
@task
def normalize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# Train RandomForest model
@task
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


# Evaluate model
@task
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Iris Model Accuracy: {acc:.2f}")
    return acc


# Prefect Flow
@flow(name="Iris_Pipeline")
def iris_pipeline():
    X, y = load_data()
    X_scaled = normalize_data(X)
    model, X_test, y_test = train_model(X_scaled, y)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    iris_pipeline()