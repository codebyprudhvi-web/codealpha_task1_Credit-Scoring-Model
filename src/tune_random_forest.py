import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cs-training.csv"
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Remove ID column
# --------------------------------------------------

df = df.drop(
    columns=["Unnamed: 0"]
)


# --------------------------------------------------
# Separate features and target
# --------------------------------------------------

X = df.drop(
    columns=["SeriousDlqin2yrs"]
)

y = df["SeriousDlqin2yrs"]


# --------------------------------------------------
# Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Random Forest pipeline
# --------------------------------------------------

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "model",
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )
    )
])


# --------------------------------------------------
# Hyperparameter search space
# --------------------------------------------------

param_distributions = {

    "model__n_estimators": [
        150,
        200,
        300
    ],

    "model__max_depth": [
        8,
        12,
        16,
        None
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ],

    "model__max_features": [
        "sqrt",
        "log2"
    ],

    "model__class_weight": [
        "balanced",
        "balanced_subsample"
    ]
}


# --------------------------------------------------
# Randomized search
# --------------------------------------------------

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


# --------------------------------------------------
# Train search
# --------------------------------------------------

print("=" * 60)
print("CODEALPHA - CREDIT SCORING MODEL")
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 60)

print("\nStarting RandomizedSearchCV...")
print("This may take some time.")


search.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Best parameters
# --------------------------------------------------

print("\n" + "=" * 60)
print("BEST PARAMETERS")
print("=" * 60)

print(search.best_params_)


print("\nBest cross-validation ROC-AUC:")
print(f"{search.best_score_:.4f}")


# --------------------------------------------------
# Evaluate tuned model
# --------------------------------------------------

best_model = search.best_estimator_

y_pred = best_model.predict(
    X_test
)

y_probability = best_model.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 60)
print("TUNED RANDOM FOREST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# Comparison with baseline
# --------------------------------------------------

print("\n" + "=" * 60)
print("BASELINE VS TUNED RANDOM FOREST")
print("=" * 60)

print("\nBaseline Random Forest:")
print("Accuracy : 0.8256")
print("Precision: 0.2357")
print("Recall   : 0.7172")
print("F1-Score : 0.3548")
print("ROC-AUC  : 0.8626")

print("\nTuned Random Forest:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


print("\nHyperparameter tuning completed successfully.")