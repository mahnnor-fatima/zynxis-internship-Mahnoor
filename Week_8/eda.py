"""
eda.py
Exploratory data analysis on the intern performance dataset.
Generates the visuals used in the capstone project report.

Run:
    python eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def main():
    df = pd.read_csv(DATA_PATH)

    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nClass balance:\n", df["Placed_Or_HighPerformer"].value_counts(normalize=True))
    print("\nEducation level counts:\n", df["Education_Level"].value_counts())

  
    fig, ax = plt.subplots(figsize=(5, 4))
    df["Placed_Or_HighPerformer"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color=["#c0392b", "#27ae60"]
    )
    ax.set_xticklabels(["Not Placed / Needs Support (0)", "High Performer / Placed (1)"], rotation=15)
    ax.set_ylabel("Count")
    ax.set_title("Target Class Balance")
    plt.tight_layout()
    plt.savefig("assets_class_balance.png", dpi=150)
    plt.close()

   
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, NUM_COLS):
        sns.histplot(df[col], kde=True, ax=ax, color="#2980b9")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig("assets_feature_distributions.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df[NUM_COLS + ["Placed_Or_HighPerformer"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("assets_correlation_heatmap.png", dpi=150)
    plt.close()
   
    fig, ax = plt.subplots(figsize=(6, 4))
    pd.crosstab(df["Education_Level"], df["Placed_Or_HighPerformer"], normalize="index").plot(
        kind="bar", stacked=True, ax=ax, color=["#c0392b", "#27ae60"]
    )
    ax.set_ylabel("Proportion")
    ax.set_title("Outcome Rate by Education Level")
    ax.legend(["Needs Support", "High Performer"], title="")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("assets_education_vs_outcome.png", dpi=150)
    plt.close()

    print("\nSaved: assets_class_balance.png, assets_feature_distributions.png, "
          "assets_correlation_heatmap.png, assets_education_vs_outcome.png")


if __name__ == "__main__":
    main()
