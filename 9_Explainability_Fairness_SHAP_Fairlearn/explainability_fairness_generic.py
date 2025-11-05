# explainability_fairness_simple_save.py
# Works on any dataset (custom CSV)
# RandomForest + SHAP + LIME + Fairness audit
# Shows and saves outputs in your folder

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import shap
from lime.lime_tabular import LimeTabularExplainer
from fairlearn.metrics import MetricFrame, selection_rate
import matplotlib.pyplot as plt
import os

# ------------------ CONFIG ------------------
CSV_PATH = "sample.csv"      # change this to your CSV file
TARGET = "gender"                  # change this to your target column
SENSITIVE_FEATURE = "gender"       # change this to your sensitive column
OUTPUT_PREFIX = "model_output"     # prefix for saved files

# ------------------ LOAD & PREPROCESS ------------------
data = pd.read_csv(CSV_PATH)
print(f"Loaded dataset: {data.shape}")

if TARGET not in data.columns:
    raise ValueError(f"Target column '{TARGET}' not found!")

if SENSITIVE_FEATURE not in data.columns:
    raise ValueError(f"Sensitive feature '{SENSITIVE_FEATURE}' not found!")

X = data.drop(columns=[TARGET])
y = data[TARGET]

# Encode categorical columns
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Scale numeric features
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

# ------------------ TRAIN-TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------ MODEL ------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")

# ------------------ SHAP EXPLAINABILITY ------------------
print("\nSHAP Explainability:")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global importance
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Feature Importance")
plt.savefig(f"{OUTPUT_PREFIX}_shap_summary.png", bbox_inches="tight")
plt.show()
print(f"SHAP summary plot saved as {OUTPUT_PREFIX}_shap_summary.png")

# ------------------ SHAP WATERFALL (Local Explanation) ------------------
print("\nExample SHAP explanation for one test sample:")

sample_idx = 0  # pick one sample to explain

# Handle binary / multiclass / regression
if isinstance(shap_values, list):  
    # If binary classification, pick positive class (1) else first class
    class_idx = 1 if len(shap_values) == 2 else 0
    shap_single = shap_values[class_idx][sample_idx]
    base_value = explainer.expected_value[class_idx]
else:
    shap_single = shap_values[sample_idx]
    base_value = explainer.expected_value

# Fix case where base_value is an array (convert to scalar)
if isinstance(base_value, (list, np.ndarray)):
    base_value = base_value[0]

# Ensure SHAP values are 1D
if shap_single.ndim > 1:
    shap_single = shap_single[:, 0]

# Build Explanation object
shap_exp = shap.Explanation(
    values=shap_single,
    base_values=base_value,
    data=X_test.iloc[sample_idx],
    feature_names=X_test.columns
)

plt.figure()
shap.plots.waterfall(shap_exp, show=False)
plt.title("SHAP Local Explanation (Sample 0)")
plt.savefig(f"{OUTPUT_PREFIX}_shap_waterfall.png", bbox_inches="tight")
plt.show()
print(f"SHAP waterfall plot saved as {OUTPUT_PREFIX}_shap_waterfall.png")



# ------------------ LIME EXPLAINABILITY ------------------
print("\nLIME Explainability:")
lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=np.unique(y_train),
    mode='classification'
)

exp = lime_explainer.explain_instance(
    data_row=X_test.iloc[0],
    predict_fn=model.predict_proba
)

exp.as_pyplot_figure()
plt.title("LIME Explanation (Top Features)")
plt.savefig(f"{OUTPUT_PREFIX}_lime_plot.png", bbox_inches="tight")
plt.show()
print(f"LIME plot saved as {OUTPUT_PREFIX}_lime_plot.png")

lime_html_path = f"{OUTPUT_PREFIX}_lime_explanation.html"
exp.save_to_file(lime_html_path)
print(f"LIME explanation HTML saved as {lime_html_path}")

# ------------------ FAIRNESS AUDIT ------------------
print("\nFairness Audit:")
sf = data[SENSITIVE_FEATURE].iloc[X_test.index]

metric_frame = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sf
)

print("\nFairness metrics by group:")
print(metric_frame.by_group)

ax = metric_frame.by_group.plot(kind="bar", subplots=True, layout=(1, 2), figsize=(10, 4))
plt.suptitle(f"Fairness metrics by {SENSITIVE_FEATURE}")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_fairness.png", bbox_inches="tight")
plt.show()
print(f"Fairness audit plot saved as {OUTPUT_PREFIX}_fairness.png")

print("\nAll tasks completed successfully! Outputs saved in current folder.")