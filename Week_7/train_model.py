"""
train_model.py
Reproduces the Week 3 model pipeline (see classification_model.ipynb) and
saves the artifacts needed by the Streamlit app: the fitted Random Forest
classifier and the fitted StandardScaler.

Run once before starting the app:
    python train_model.py
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

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

    X = df.drop(columns=["Intern_ID", "Placed_Or_HighPerformer"])
    y = df["Placed_Or_HighPerformer"]

    # One-hot encode Education_Level (drop_first=True, same as the notebook)
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

    model = RandomForestClassifier(
        random_state=42, n_estimators=100, max_depth=7, min_samples_split=4
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    joblib.dump(model, "best_model_random_forest.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feature_names, "feature_names.pkl")

    print("\nSaved artifacts:")
    print("  - best_model_random_forest.pkl")
    print("  - scaler.pkl")
    print("  - feature_names.pkl")


if __name__ == "__main__":
    main()
