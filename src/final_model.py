import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/fraud_oracle.csv")

print("Dataset:", df.shape)


# ============================================================
# 2. REMOVE IDENTIFIER
# ============================================================

df = df.drop(columns=["PolicyNumber"])


# ============================================================
# 3. FEATURES / TARGET
# ============================================================

X = df.drop(columns=["FraudFound_P"])
y = df["FraudFound_P"]


# ============================================================
# 4. COLUMN TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full
)

print("\nTrain:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# 7. CLASS IMBALANCE
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nNegative samples:", negative)
print("Positive samples:", positive)
print(
    "Scale pos weight:",
    round(scale_pos_weight, 2)
)


# ============================================================
# 8. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 9. XGBOOST
# ============================================================

model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,

    scale_pos_weight=scale_pos_weight,

    eval_metric="logloss",

    random_state=42,
    n_jobs=-1
)


# ============================================================
# 10. PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 11. TRAIN
# ============================================================

print("\nTraining XGBoost...")

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# 12. VALIDATION PREDICTIONS
# ============================================================

val_prob = pipeline.predict_proba(
    X_val
)[:, 1]


# ============================================================
# 13. FIND BEST THRESHOLD ON VALIDATION SET
# ============================================================

thresholds = np.arange(
    0.05,
    0.51,
    0.01
)

best_threshold = 0.5
best_f1 = 0

print("\nFinding best threshold...")

for threshold in thresholds:

    val_pred = (
        val_prob >= threshold
    ).astype(int)

    f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print(
    "\nBest validation threshold:",
    round(best_threshold, 2)
)

print(
    "Validation F1:",
    round(best_f1, 4)
)


# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================

test_prob = pipeline.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_prob >= best_threshold
).astype(int)


# ============================================================
# 15. METRICS
# ============================================================

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_prob
)

average_precision = average_precision_score(
    y_test,
    test_prob
)


# ============================================================
# 16. RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    "Threshold:",
    round(best_threshold, 2)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F1 Score:",
    round(f1, 4)
)

print(
    "ROC-AUC:",
    round(roc_auc, 4)
)

print(
    "Average Precision:",
    round(average_precision, 4)
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_pred
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_pred,
        zero_division=0
    )
)


# ============================================================
# 17. SAVE MODEL
# ============================================================

import os

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    pipeline,
    "models/fraud_model.pkl"
)

print(
    "\nModel saved to:",
    "models/fraud_model.pkl"
)


# ============================================================
# 18. SAVE TEST PREDICTIONS
# ============================================================

results = X_test.copy()

results["Actual_Fraud"] = y_test.values
results["Fraud_Probability"] = test_prob
results["Predicted_Fraud"] = test_pred

results.to_csv(
    "models/test_predictions.csv",
    index=False
)

print(
    "Predictions saved to:",
    "models/test_predictions.csv"
)