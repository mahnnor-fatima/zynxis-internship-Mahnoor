import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set overall style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

output_dir = r"c:\Users\mahno\OneDrive\Desktop\internship\Week_1"
train_path = os.path.join(output_dir, "train_data.csv")
test_path = os.path.join(output_dir, "test_data.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
df = pd.concat([train_df, test_df], ignore_index=True)

palette_survival = {0: "#E74C3C", 1: "#2ECC71"} # Red for Deceased, Green for Survived
palette_gender = {0: "#9B59B6", 1: "#3498DB"}    # Purple for Female, Blue for Male

# 1. Target Class Balance
plt.figure(figsize=(7, 5))
ax = sns.countplot(x='Survived', data=df, palette=palette_survival)
plt.title("Figure 1: Overall Passenger Survival Distribution", fontweight='bold', pad=15)
plt.xlabel("Survival Status (0 = Deceased, 1 = Survived)")
plt.ylabel("Passenger Count")
plt.xticks([0, 1], ['Deceased (0)', 'Survived (1)'])

total = len(df)
for p in ax.patches:
    height = p.get_height()
    percentage = f"{100 * height / total:.1f}% ({height})"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2., height / 2),
                ha='center', va='center', fontsize=12, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig1_survival_balance.png"), dpi=300)
plt.close()

# 2. Survival by Gender
plt.figure(figsize=(8, 5))
df['Gender_Label'] = df['Sex'].map({0: 'Female', 1: 'Male'})
ax = sns.countplot(x='Gender_Label', hue='Survived', data=df, palette=palette_survival)
plt.title("Figure 2: Survival Breakdown by Gender", fontweight='bold', pad=15)
plt.xlabel("Gender")
plt.ylabel("Passenger Count")
plt.legend(title='Status', labels=['Deceased', 'Survived'])

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height + 5),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig2_survival_by_gender.png"), dpi=300)
plt.close()

# 3. Passenger Class Survival
plt.figure(figsize=(8, 5))
df['Pclass'] = np.where(df['Pclass_1'] == 1, '1st Class',
               np.where(df['Pclass_2'] == 1, '2nd Class', '3rd Class'))

pclass_order = ['1st Class', '2nd Class', '3rd Class']
ax = sns.barplot(x='Pclass', y='Survived', data=df, order=pclass_order, ci=None, palette="Blues_r")
plt.title("Figure 3: Survival Rate by Socio-Economic Class", fontweight='bold', pad=15)
plt.xlabel("Ticket Class")
plt.ylabel("Survival Rate (0.0 to 1.0)")
plt.ylim(0, 1.0)

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f"{height*100:.1f}%", (p.get_x() + p.get_width() / 2., height + 0.02),
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2C3E50')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig3_pclass_survival.png"), dpi=300)
plt.close()

# 4. Age & Fare Distribution by Survival
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.kdeplot(data=df, x='Age', hue='Survived', palette=palette_survival, common_norm=False, fill=True, alpha=0.4, ax=axes[0])
axes[0].set_title("Age Distribution by Survival (Normalized)", fontweight='bold')
axes[0].set_xlabel("Scaled Age")

sns.kdeplot(data=df, x='Fare', hue='Survived', palette=palette_survival, common_norm=False, fill=True, alpha=0.4, ax=axes[1])
axes[1].set_title("Fare Distribution by Survival (Normalized)", fontweight='bold')
axes[1].set_xlabel("Scaled Ticket Fare")

plt.suptitle("Figure 4: Age & Fare Distribution Patterns", fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig4_age_fare_dist.png"), dpi=300)
plt.close()

# 5. Correlation Heatmap
plt.figure(figsize=(10, 8))
cols_for_corr = ['Survived', 'Sex', 'Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Family_size', 'Title_1', 'Title_2', 'Emb_3']
corr_matrix = df[cols_for_corr].corr()

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.6, vmax=0.6, linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Figure 5: Feature Correlation Matrix with Survival", fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig5_correlation_heatmap.png"), dpi=300)
plt.close()

# 6. Family Size vs Survival
plt.figure(figsize=(8, 5))
ax = sns.barplot(x='Family_size', y='Survived', data=df, ci=None, palette="Purples")
plt.title("Figure 6: Survival Probability by Family Size (Scaled)", fontweight='bold', pad=15)
plt.xlabel("Scaled Family Size")
plt.ylabel("Survival Rate")

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f"{height*100:.1f}%", (p.get_x() + p.get_width() / 2., height + 0.02),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig6_family_survival.png"), dpi=300)
plt.close()

print("All 6 figures generated successfully!")
