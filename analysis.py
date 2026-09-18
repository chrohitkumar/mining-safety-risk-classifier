"""
Mine Shift Safety Risk Prediction
----------------------------------
Predicts whether a given underground mining shift is likely to have a
safety incident, based on operational parameters available before/during
the shift. Goal: flag high-risk shifts in advance so supervisors can act
(extra ventilation checks, reduced shift length, pairing with experienced
workers, etc.)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

OUT = "/home/claude/mine_safety_risk"
sns.set_style("whitegrid")

# ---------- 1. Load data ----------
df = pd.read_csv(f"{OUT}/mine_shift_safety_data.csv")
print("Shape:", df.shape)
print(df.describe().T)

FEATURES = [
    "shift_duration_hrs", "ventilation_rate_m3_per_min", "methane_level_ppm",
    "machinery_age_years", "avg_worker_experience_yrs", "workforce_size",
    "depth_m", "temperature_c", "incidents_last_30_days"
]
TARGET = "incident_occurred"

# ---------- 2. EDA: correlation heatmap ----------
plt.figure(figsize=(9, 7))
corr = df[FEATURES + [TARGET]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap - Shift Safety Factors")
plt.tight_layout()
plt.savefig(f"{OUT}/01_correlation_heatmap.png", dpi=150)
plt.close()

# ---------- 3. EDA: ventilation vs incident rate ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
df["vent_bucket"] = pd.cut(df["ventilation_rate_m3_per_min"], bins=5)
vent_rate = df.groupby("vent_bucket", observed=True)[TARGET].mean()
vent_rate.plot(kind="bar", ax=axes[0], color="#2c7fb8")
axes[0].set_title("Incident Rate by Ventilation Rate")
axes[0].set_ylabel("Incident Rate")
axes[0].set_xlabel("Ventilation Rate (m3/min), binned")

df["exp_bucket"] = pd.cut(df["avg_worker_experience_yrs"], bins=5)
exp_rate = df.groupby("exp_bucket", observed=True)[TARGET].mean()
exp_rate.plot(kind="bar", ax=axes[1], color="#de2d26")
axes[1].set_title("Incident Rate by Workforce Experience")
axes[1].set_ylabel("Incident Rate")
axes[1].set_xlabel("Avg Worker Experience (yrs), binned")
plt.tight_layout()
plt.savefig(f"{OUT}/02_eda_key_factors.png", dpi=150)
plt.close()

# ---------- 4. Train / test split ----------
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------- 5. Model 1: Logistic Regression (interpretable baseline) ----------
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_s, y_train)
y_pred_lr = log_reg.predict(X_test_s)
y_prob_lr = log_reg.predict_proba(X_test_s)[:, 1]

# ---------- 6. Model 2: Random Forest (stronger, still explainable via feature importance) ----------
rf = RandomForestClassifier(
    n_estimators=300, max_depth=6, class_weight="balanced", random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]


def report(name, y_true, y_pred, y_prob):
    print(f"\n--- {name} ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.3f}")
    print(f"F1       : {f1_score(y_true, y_pred):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_prob):.3f}")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


metrics_lr = report("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
metrics_rf = report("Random Forest", y_test, y_pred_rf, y_prob_rf)

# ---------- 7. Confusion matrices ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, y_pred) in zip(axes, [("Logistic Regression", y_pred_lr), ("Random Forest", y_pred_rf)]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Incident", "Incident"],
                yticklabels=["No Incident", "Incident"])
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT}/03_confusion_matrices.png", dpi=150)
plt.close()

# ---------- 8. ROC curves ----------
plt.figure(figsize=(6.5, 5.5))
for name, y_prob in [("Logistic Regression", y_prob_lr), ("Random Forest", y_prob_rf)]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Incident Prediction")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/04_roc_curve.png", dpi=150)
plt.close()

# ---------- 9. Feature importance (Random Forest) ----------
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
plt.figure(figsize=(8, 5.5))
importances.plot(kind="barh", color="#31a354")
plt.title("Feature Importance - What Drives Incident Risk")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/05_feature_importance.png", dpi=150)
plt.close()

print("\nTop risk factors (Random Forest importance):")
print(importances.sort_values(ascending=False))

# ---------- 10. Save a summary text file ----------
with open(f"{OUT}/results_summary.txt", "w") as f:
    f.write("MINE SHIFT SAFETY RISK PREDICTION - RESULTS SUMMARY\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Dataset: {len(df)} shift records, {df[TARGET].mean():.1%} incident rate\n\n")
    f.write("Logistic Regression:\n")
    for k, v in metrics_lr.items():
        f.write(f"  {k}: {v:.3f}\n")
    f.write("\nRandom Forest:\n")
    for k, v in metrics_rf.items():
        f.write(f"  {k}: {v:.3f}\n")
    f.write("\nTop 5 risk factors (Random Forest importance):\n")
    for feat, imp in importances.sort_values(ascending=False).head(5).items():
        f.write(f"  {feat}: {imp:.3f}\n")

print("\nAll plots and summary saved.")
