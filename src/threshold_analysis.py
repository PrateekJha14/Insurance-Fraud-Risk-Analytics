import pandas as pd
import numpy as np

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
    average_precision_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/fraud_oracle.csv")

# Remove identifier
df = df.drop(columns=["PolicyNumber"])

X = df.drop(columns=["FraudFound_P"])
y = df["FraudFound_P"]


# ============================================================
# 2. COLUMN TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. PREPROCESSING
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
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 5. XGBOOST
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 6. TRAIN
# ============================================================

print("Training XGBoost...")

pipeline.fit(X_train, y_train)

y_prob = pipeline.predict_proba(X_test)[:, 1]


# ============================================================
# 7. PROBABILITY METRICS
# ============================================================

roc_auc = roc_auc_score(y_test, y_prob)
average_precision = average_precision_score(y_test, y_prob)

print("\nROC-AUC:", round(roc_auc, 4))
print("Average Precision:", round(average_precision, 4))


# ============================================================
# 8. TEST DIFFERENT THRESHOLDS
# ============================================================

thresholds = np.arange(0.05, 0.96, 0.05)

results = []

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

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

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


results_df = pd.DataFrame(results)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\nThreshold Analysis:")
print(
    results_df.to_string(index=False)
)


# ============================================================
# 10. BEST F1 THRESHOLD
# ============================================================

best_row = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n" + "=" * 60)
print("BEST F1 THRESHOLD")
print("=" * 60)

print(
    f"Threshold : {best_row['Threshold']:.2f}"
)

print(
    f"Precision : {best_row['Precision']:.4f}"
)

print(
    f"Recall    : {best_row['Recall']:.4f}"
)

print(
    f"F1 Score  : {best_row['F1']:.4f}"
)