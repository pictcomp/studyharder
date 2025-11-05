# automate_prefect.py
# Automates preprocessing and training using Prefect with Wine Quality dataset

from prefect import flow, task
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
@task
def load_data():
    data = pd.read_csv(r"FILEPATH/WineQT.csv")
    X = data.drop("quality", axis=1)
    y = data["quality"]
    return X, y


# Preprocess data (normalize)
@task
def normalize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# Train model
@task
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


# Evaluate model
@task
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.2f}")
    return acc


# Prefect flow
@flow(name="WineQuality_Pipeline")
def wine_pipeline():
    X, y = load_data()
    X_scaled = normalize_data(X)
    model, X_test, y_test = train_model(X_scaled, y)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    wine_pipeline()
