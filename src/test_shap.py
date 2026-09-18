import pandas as pd
import numpy as np
import joblib
import shap


# ============================================================
# LOAD MODEL + TEST PREDICTIONS
# ============================================================

model_pipeline = joblib.load(
    "models/fraud_model.pkl"
)

predictions = pd.read_csv(
    "models/test_predictions.csv"
)


# ============================================================
# SELECT CLAIM
# ============================================================

claim_index = 7

claim_row = predictions.iloc[
    claim_index
]


# ============================================================
# PREPARE CLAIM FOR MODEL
# ============================================================

# Remove columns that were not model features
claim = claim_row.drop(
    labels=[
        "Actual_Fraud",
        "Fraud_Probability",
        "Predicted_Fraud"
    ]
).to_frame().T


# ============================================================
# GET PIPELINE COMPONENTS
# ============================================================

preprocessor = (
    model_pipeline
    .named_steps["preprocessor"]
)

xgb_model = (
    model_pipeline
    .named_steps["model"]
)


# ============================================================
# TRANSFORM CLAIM
# ============================================================

X_transformed = (
    preprocessor.transform(
        claim
    )
)


# ============================================================
# GET FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    xgb_model
)

shap_values = explainer.shap_values(
    X_transformed
)


# ============================================================
# CREATE RESULTS
# ============================================================

values = shap_values[0]

result = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Value": values,
    "Impact": np.abs(values)
})


result = result.sort_values(
    "Impact",
    ascending=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("SHAP EXPLANATION FOR CLAIM", claim_index)
print("=" * 60)

print()

print(
    result[
        [
            "Feature",
            "SHAP_Value"
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# COMPARE MODEL PROBABILITY
# ============================================================

model_probability = (
    model_pipeline
    .predict_proba(claim)[:, 1][0]
)

saved_probability = (
    claim_row["Fraud_Probability"]
)


print()
print("=" * 60)
print("PROBABILITY CHECK")
print("=" * 60)

print(
    f"Saved prediction : {saved_probability:.6f}"
)

print(
    f"Model prediction : {model_probability:.6f}"
)

print(
    f"Risk score       : {model_probability * 100:.2f}%"
)