"""
evaluate.py
Detailed evaluation of the trained Random Forest model: confusion matrix,
ROC curve, and feature importance. Generates the visuals used in the
capstone project report and slides.

Run after train_model.py:
    python evaluate.py
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report,
)

sns.set_style("whitegrid")

DATA_PATH = "zynxis_intern_performance.csv"
NUM_COLS = [
    "Technical_Score",
    "Project_Completion_Rate",
    "Attendance_Punctuality",
    "Soft_Skills_Rating",
    "Code_Review_Score",
    "Prior_Experience_Months",
]
CAT_COL = "Education_Level"


def main():
    df = pd.read_csv(DATA_PATH)
    model = joblib.load("best_model_random_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")

    X = df.drop(columns=["Intern_ID", "Placed_Or_HighPerformer"])
    y = df["Placed_Or_HighPerformer"]
    X_encoded = pd.get_dummies(X, columns=[CAT_COL], drop_first=True)[feature_names]

    _, X_test, _, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )
    X_test_scaled = X_test.copy()
    X_test_scaled[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("=== Test Set Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\n", classification_report(y_test, y_pred, target_names=["Needs Support", "High Performer"]))

    with open("evaluation_metrics.txt", "w") as f:
        f.write("=== Test Set Metrics ===\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"ROC-AUC:   {auc:.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=["Needs Support", "High Performer"]))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Needs Support", "High Performer"],
        yticklabels=["Needs Support", "High Performer"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("assets_confusion_matrix.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color="#2980b9", linewidth=2, label=f"ROC curve (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("assets_roc_curve.png", dpi=150)
    plt.close()

    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    importances.plot(kind="barh", ax=ax, color="#16a085")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig("assets_feature_importance.png", dpi=150)
    plt.close()

    print("\nSaved: assets_confusion_matrix.png, assets_roc_curve.png, "
          "assets_feature_importance.png, evaluation_metrics.txt")


if __name__ == "__main__":
    main()
