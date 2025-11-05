# explainability_fairness_toy.py
# Uses SHAP, LIME, and Fairlearn to explain RandomForest model on Iris dataset

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
from lime.lime_tabular import LimeTabularExplainer
from fairlearn.metrics import MetricFrame, selection_rate
import matplotlib.pyplot as plt

# Load dataset
def load_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    # Add synthetic sensitive feature
    gender = np.where(np.arange(len(X)) % 2 == 0, "male", "female")
    return X, y, gender, iris.feature_names

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

# SHAP explainability
def explain_with_shap(model, X_test, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  # list of arrays (n_classes, n_samples, n_features)

    # Ensure correct shape handling (take mean absolute across classes)
    if isinstance(shap_values, list):
        shap_values_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_values_mean = np.abs(shap_values)

    # Double-check shapes
    print("SHAP values shape:", shap_values_mean.shape)
    print("X_test shape:", X_test.shape)

    # Plot summary (works for multiclass)
    shap.summary_plot(
        shap_values_mean,
        features=X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance (Iris Dataset)")
    plt.savefig("shap_summary_iris.png", bbox_inches="tight")
    plt.close()
    print("SHAP summary plot saved as shap_summary_iris.png")


# LIME explainability
def explain_with_lime(model, X_train, X_test, feature_names):
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        mode="classification"
    )
    exp = explainer.explain_instance(X_test[0], model.predict_proba)
    exp.save_to_file("lime_explanation_iris.html")
    print("LIME explanation saved as lime_explanation_iris.html")

# Fairness audit
def audit_fairness(model, X_test, y_test, gender):
    y_pred = model.predict(X_test)
    metric_frame = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=gender[:len(y_test)]
    )
    print("\nFairness Metrics by Gender (Iris Dataset):")
    print(metric_frame.by_group)

    metric_frame.by_group.plot(kind="bar", subplots=True, layout=(1, 2), figsize=(8, 4))
    plt.title("Fairness Audit by Gender (Iris Dataset)")
    plt.savefig("fairness_audit_iris.png")
    print("Fairness audit plot saved as fairness_audit_iris.png")

# Run everything
def main():
    X, y, gender, feature_names = load_data()
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    explain_with_shap(model, X_test, feature_names)
    explain_with_lime(model, X_train, X_test, feature_names)
    audit_fairness(model, X_test, y_test, gender)
    print("Completed explainability and fairness analysis for Iris Dataset!")

if __name__ == "__main__":
    main()
