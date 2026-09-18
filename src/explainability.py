import os

import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/fraud_oracle.csv")

df = df.drop(columns=["PolicyNumber"])

X = df.drop(columns=["FraudFound_P"])
y = df["FraudFound_P"]


# ============================================================
# 2. RECREATE SAME TEST SPLIT
# ============================================================

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. LOAD TRAINED PIPELINE
# ============================================================

import joblib

pipeline = joblib.load(
    "models/fraud_model.pkl"
)

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]


# ============================================================
# 4. TRANSFORM TEST DATA
# ============================================================

X_test_transformed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

X_test_transformed = pd.DataFrame(
    X_test_transformed,
    columns=feature_names,
    index=X_test.index
)


# ============================================================
# 5. XGBOOST FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP 20 FEATURES")
print("=" * 60)

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 6. SAVE FEATURE IMPORTANCE
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

importance_df.to_csv(
    "models/feature_importance.csv",
    index=False
)


# ============================================================
# 7. PLOT TOP FEATURES
# ============================================================

top_features = importance_df.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top Features Influencing Fraud Risk")

plt.tight_layout()

plt.savefig(
    "models/feature_importance.png",
    dpi=200
)

plt.show()


# ============================================================
# 8. SHAP ANALYSIS
# ============================================================

print("\nCalculating SHAP values...")

# Use a sample to keep computation fast
sample_size = min(
    500,
    len(X_test_transformed)
)

X_sample = X_test_transformed.sample(
    n=sample_size,
    random_state=42
)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_sample
)


# ============================================================
# 9. SHAP SUMMARY PLOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    show=False,
    max_display=15
)

plt.tight_layout()

plt.savefig(
    "models/shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print("models/feature_importance.csv")
print("models/feature_importance.png")
print("models/shap_summary.png")