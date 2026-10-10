import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cs-training.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Remove the dataset index column
df = df.drop(columns=["Unnamed: 0"])

# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("=" * 60)
print("CODEALPHA - CREDIT SCORING MODEL")
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

# --------------------------------------------------
# Target distribution
# --------------------------------------------------

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

target_counts = df["SeriousDlqin2yrs"].value_counts()
target_percent = df["SeriousDlqin2yrs"].value_counts(normalize=True) * 100

print(target_counts)

print("\nPercentage:")
print(target_percent.round(2))

# --------------------------------------------------
# Plot 1 - Target distribution
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="SeriousDlqin2yrs"
)

plt.title("Target Distribution")
plt.xlabel("Serious Delinquency Within 2 Years")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "target_distribution.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# Numerical columns
# --------------------------------------------------

numerical_columns = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]

# --------------------------------------------------
# Plot 2 - Feature distributions
# --------------------------------------------------

df[numerical_columns].hist(
    figsize=(15, 12),
    bins=30
)

plt.suptitle(
    "Feature Distributions",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_distributions.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# Plot 3 - Boxplots for important numerical features
# --------------------------------------------------

important_features = [
    "age",
    "MonthlyIncome",
    "DebtRatio",
    "RevolvingUtilizationOfUnsecuredLines"
]

for feature in important_features:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="SeriousDlqin2yrs",
        y=feature
    )

    plt.title(
        f"{feature} vs Serious Delinquency"
    )

    plt.xlabel(
        "Serious Delinquency Within 2 Years"
    )

    plt.ylabel(feature)

    plt.tight_layout()

    safe_name = feature.replace("/", "_")

    plt.savefig(
        OUTPUT_DIR / f"boxplot_{safe_name}.png",
        dpi=300
    )

    plt.show()

# --------------------------------------------------
# Plot 4 - Correlation matrix
# --------------------------------------------------

plt.figure(figsize=(12, 9))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_matrix.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# Mean feature values by target
# --------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE MEANS BY TARGET")
print("=" * 60)

mean_by_target = df.groupby(
    "SeriousDlqin2yrs"
)[numerical_columns].mean().T

print(mean_by_target.round(2))

# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()

missing = missing[missing > 0].sort_values(
    ascending=False
)

print(missing)

print("\nEDA completed successfully.")

print("\nGenerated files:")
print("- outputs/target_distribution.png")
print("- outputs/feature_distributions.png")
print("- outputs/boxplot_age.png")
print("- outputs/boxplot_MonthlyIncome.png")
print("- outputs/boxplot_DebtRatio.png")
print("- outputs/boxplot_RevolvingUtilizationOfUnsecuredLines.png")
print("- outputs/correlation_matrix.png")