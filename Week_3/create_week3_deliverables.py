import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

# ---------------------------------------------------------
# Step 1: Generate Synthetic Dataset
# ---------------------------------------------------------
np.random.seed(42)
n_samples = 500

intern_ids = [f"ZYN-2026-{i+1:03d}" for i in range(n_samples)]
education_levels = np.random.choice(["High_School", "Undergrad", "Grad"], size=n_samples, p=[0.20, 0.55, 0.25])

tech_score = np.clip(np.random.normal(72, 14, n_samples), 10, 100)
project_rate = np.clip(np.random.normal(82, 12, n_samples), 0, 100)
attendance = np.clip(np.random.normal(88, 10, n_samples), 0, 100)
soft_skills = np.clip(np.random.normal(3.8, 0.8, n_samples), 1.0, 5.0)
code_review = np.clip(np.random.normal(7.2, 1.6, n_samples), 1.0, 10.0)
prior_exp = np.random.poisson(lam=5, size=n_samples)
prior_exp = np.clip(prior_exp, 0, 24)

# Calculate continuous success score and convert to binary target with ~60% positive class rate
score = (
    0.25 * tech_score +
    0.20 * project_rate +
    0.10 * attendance +
    3.5 * soft_skills +
    2.5 * code_review +
    0.8 * prior_exp +
    0.02 * (tech_score * code_review)
)
# Add some noise
noise = np.random.normal(0, 10, n_samples)
final_score = score + noise

threshold = np.percentile(final_score, 38)  # ~62% positive class, ~38% negative class
target = (final_score > threshold).astype(int)

df = pd.DataFrame({
    'Intern_ID': intern_ids,
    'Technical_Score': np.round(tech_score, 1),
    'Project_Completion_Rate': np.round(project_rate, 1),
    'Attendance_Punctuality': np.round(attendance, 1),
    'Soft_Skills_Rating': np.round(soft_skills, 1),
    'Code_Review_Score': np.round(code_review, 1),
    'Prior_Experience_Months': prior_exp,
    'Education_Level': education_levels,
    'Placed_Or_HighPerformer': target
})

output_dir = r"c:\Users\mahno\OneDrive\Desktop\internship\Week_3"
dataset_path = os.path.join(output_dir, "zynxis_intern_performance.csv")
df.to_csv(dataset_path, index=False)
print(f"Dataset successfully saved to {dataset_path} with shape {df.shape}")

# ---------------------------------------------------------
# Step 2: Run Machine Learning Pipeline
# ---------------------------------------------------------
X = df.drop(columns=['Intern_ID', 'Placed_Or_HighPerformer'])
y = df['Placed_Or_HighPerformer']

# Categorical vs Numerical features
num_features = ['Technical_Score', 'Project_Completion_Rate', 'Attendance_Punctuality', 
                'Soft_Skills_Rating', 'Code_Review_Score', 'Prior_Experience_Months']
cat_features = ['Education_Level']

# One-hot encode education level
df_processed = pd.get_dummies(X, columns=cat_features, drop_first=True)
feature_names = list(df_processed.columns)

# Train-Test Split (80/20 Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    df_processed, y, test_size=0.20, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])
X_test_scaled[num_features] = scaler.transform(X_test[num_features])

# Save Scaler
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

# Define Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=6),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=7, min_samples_split=4)
}

results = {}
confusion_matrices = {}
roc_data = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'ROC-AUC': roc_auc
    }
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, roc_auc)

# Save best model (Random Forest)
joblib.dump(models["Random Forest"], os.path.join(output_dir, "best_model_random_forest.pkl"))

# Convert results to DataFrame
metrics_df = pd.DataFrame(results).T.round(4)
comparison_table_path = os.path.join(output_dir, "model_comparison_table.csv")
metrics_df.to_csv(comparison_table_path)
print("Model Comparison Metrics:")
print(metrics_df)

# ---------------------------------------------------------
# Step 3: Generate and Save Visualizations
# ---------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.size'] = 11

# 1. Model Metrics Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
metrics_df.plot(kind='bar', ax=ax, width=0.8, colormap='viridis')
plt.title("Zynxis Model Performance Comparison Across Key Evaluation Metrics", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
plt.xlabel("Classification Models", fontsize=12)
plt.xticks(rotation=0, fontweight='bold')
plt.ylim(0.5, 1.02)
plt.legend(title="Metrics", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
metrics_fig_path = os.path.join(output_dir, "model_metrics_comparison.png")
plt.savefig(metrics_fig_path, dpi=300)
plt.close()

# 2. Confusion Matrices (Side by Side)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, (name, cm) in enumerate(confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                xticklabels=['Not Placed (0)', 'Placed (1)'],
                yticklabels=['Not Placed (0)', 'Placed (1)'])
    axes[idx].set_title(f"{name}\nAcc: {results[name]['Accuracy']:.3f} | F1: {results[name]['F1 Score']:.3f}", fontsize=12, fontweight='bold')
    axes[idx].set_xlabel("Predicted Label", fontsize=10)
    axes[idx].set_ylabel("True Label", fontsize=10)

plt.suptitle("Confusion Matrices for Classification Models", fontsize=16, fontweight='bold', y=1.03)
plt.tight_layout()
cm_fig_path = os.path.join(output_dir, "confusion_matrices.png")
plt.savefig(cm_fig_path, dpi=300)
plt.close()

# 3. ROC Curves
plt.figure(figsize=(9, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for idx, (name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=colors[idx], lw=2.5)

plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.50)', lw=1.5)
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight='bold', pad=15)
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()
roc_fig_path = os.path.join(output_dir, "roc_curves.png")
plt.savefig(roc_fig_path, dpi=300)
plt.close()

# 4. Feature Importances (Random Forest)
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='magma')
plt.title("Random Forest Feature Importances (Key Drivers of Success)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Relative Feature Importance Score", fontsize=12)
plt.ylabel("Feature Name", fontsize=12)
plt.tight_layout()
imp_fig_path = os.path.join(output_dir, "feature_importance.png")
plt.savefig(imp_fig_path, dpi=300)
plt.close()

print("All charts and model artifacts generated successfully!")
