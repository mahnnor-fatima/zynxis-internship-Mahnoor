"""
train_model.py
Trains the final Random Forest model plus two baseline models used for
comparison in the app (Logistic Regression, Decision Tree). Saves all
artifacts the app needs.

Run once before starting the app:
    python train_model.py
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

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


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Intern_ID", "Placed_Or_HighPerformer"])
    y = df["Placed_Or_HighPerformer"]

    X_encoded = pd.get_dummies(X, columns=[CAT_COL], drop_first=True)
    feature_names = list(X_encoded.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test_scaled[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

    rf = RandomForestClassifier(
        random_state=42, n_estimators=100, max_depth=7, min_samples_split=4
    )
    rf.fit(X_train_scaled, y_train)
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg.fit(X_train_scaled, y_train)

    dtree = DecisionTreeClassifier(random_state=42, max_depth=5)
    dtree.fit(X_train_scaled, y_train)

    results = [
        evaluate("Random Forest (deployed)", rf, X_test_scaled, y_test),
        evaluate("Logistic Regression", logreg, X_test_scaled, y_test),
        evaluate("Decision Tree", dtree, X_test_scaled, y_test),
    ]
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    joblib.dump(rf, "best_model_random_forest.pkl")
    joblib.dump(logreg, "model_logistic_regression.pkl")
    joblib.dump(dtree, "model_decision_tree.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feature_names, "feature_names.pkl")
    results_df.to_csv("model_comparison.csv", index=False)

    print("\nSaved artifacts:")
    print(" best_model_random_forest.pkl  (deployed model)")
    print(" model_logistic_regression.pkl (comparison)")
    print("model_decision_tree.pkl       (comparison)")
    print("scaler.pkl")
    print("feature_names.pkl")
    print("model_comparison.csv")


if __name__ == "__main__":
    main()
