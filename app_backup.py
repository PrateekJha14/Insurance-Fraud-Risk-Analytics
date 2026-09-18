import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Insurance Fraud Risk Analytics",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    predictions = pd.read_csv(
        "models/test_predictions.csv"
    )

    importance = pd.read_csv(
        "models/feature_importance.csv"
    )

    original_data = pd.read_csv(
        "data/fraud_oracle.csv"
    )

    return predictions, importance, original_data


predictions, importance, original_data = load_data()


# ============================================================
# TITLE
# ============================================================

st.title("🛡️ Insurance Fraud Risk Analytics")

st.markdown(
    """
    **Machine-learning powered insurance claim risk screening**

    This dashboard analyzes insurance claims using an XGBoost
    classification model and provides fraud-risk scores to help
    prioritize potentially high-risk claims for further investigation.
    """
)

st.divider()


# ============================================================
# RISK SCORE
# ============================================================

predictions["Risk Score"] = (
    predictions["Fraud_Probability"] * 100
)


def risk_category(score):

    if score >= 70:
        return "High Risk"
    elif score >= 30:
        return "Medium Risk"
    else:
        return "Low Risk"


predictions["Risk Level"] = predictions["Risk Score"].apply(
    risk_category
)


# ============================================================
# KPI SECTION
# ============================================================

total_claims = len(original_data)

actual_fraud = int(
    original_data["FraudFound_P"].sum()
)

fraud_rate = (
    actual_fraud / total_claims
) * 100

roc_auc = roc_auc_score(
    predictions["Actual_Fraud"],
    predictions["Fraud_Probability"]
)

avg_precision = average_precision_score(
    predictions["Actual_Fraud"],
    predictions["Fraud_Probability"]
)

threshold = 0.49

predicted_fraud = (
    predictions["Fraud_Probability"] >= threshold
).astype(int)

precision = precision_score(
    predictions["Actual_Fraud"],
    predicted_fraud,
    zero_division=0
)

recall = recall_score(
    predictions["Actual_Fraud"],
    predicted_fraud,
    zero_division=0
)

f1 = f1_score(
    predictions["Actual_Fraud"],
    predicted_fraud,
    zero_division=0
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Claims",
    f"{total_claims:,}"
)

col2.metric(
    "Fraud Claims",
    f"{actual_fraud:,}"
)

col3.metric(
    "Fraud Rate",
    f"{fraud_rate:.2f}%"
)

col4.metric(
    "ROC-AUC",
    f"{roc_auc:.3f}"
)


st.divider()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Risk Filters")

risk_filter = st.sidebar.multiselect(
    "Risk Level",
    options=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ],
    default=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_predictions = predictions[
    predictions["Risk Level"].isin(risk_filter)
]


# ============================================================
# RISK DISTRIBUTION
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader("Claim Risk Distribution")

    risk_counts = predictions[
        "Risk Level"
    ].value_counts()

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        risk_counts.index,
        risk_counts.values
    )

    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Claims")
    ax.set_title("Claims by Risk Level")

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


with col2:

    st.subheader("Actual Fraud Distribution")

    fraud_counts = original_data[
        "FraudFound_P"
    ].value_counts()

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        ["Legitimate", "Fraud"],
        [
            fraud_counts.get(0, 0),
            fraud_counts.get(1, 0)
        ]
    )

    ax.set_ylabel("Number of Claims")
    ax.set_title("Actual Claim Distribution")

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# RISK SCORE DISTRIBUTION
# ============================================================

st.subheader("Fraud Risk Score Distribution")

fig, ax = plt.subplots(
    figsize=(12, 4)
)

ax.hist(
    predictions["Risk Score"],
    bins=30
)

ax.axvline(
    threshold * 100,
    linestyle="--",
    label=f"Investigation Threshold ({threshold:.2f})"
)

ax.set_xlabel("Fraud Risk Score")
ax.set_ylabel("Number of Claims")
ax.set_title("Distribution of Model Risk Scores")

ax.legend()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# TOP FEATURES
# ============================================================

st.subheader("Top Features Influencing Fraud Risk")

top_features = importance.head(12).copy()

top_features["Feature"] = (
    top_features["Feature"]
    .str.replace("cat__", "", regex=False)
    .str.replace("num__", "", regex=False)
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

ax.set_xlabel("Model Feature Importance")
ax.set_ylabel("Feature")

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# HIGH RISK CLAIMS
# ============================================================

st.subheader("High-Risk Claims for Investigation")

high_risk = predictions[
    predictions["Risk Level"] == "High Risk"
].copy()

display_columns = [
    "Fraud_Probability",
    "Risk Score",
    "Actual_Fraud",
    "Predicted_Fraud"
]

high_risk_display = high_risk[
    display_columns
].copy()

high_risk_display = high_risk_display.sort_values(
    by="Risk Score",
    ascending=False
)

high_risk_display["Fraud_Probability"] = (
    high_risk_display["Fraud_Probability"]
    .round(3)
)

high_risk_display["Risk Score"] = (
    high_risk_display["Risk Score"]
    .round(1)
)

st.dataframe(
    high_risk_display.head(25),
    use_container_width=True
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.subheader("Model Performance")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric(
    "Precision",
    f"{precision:.3f}"
)

metric_col2.metric(
    "Recall",
    f"{recall:.3f}"
)

metric_col3.metric(
    "F1 Score",
    f"{f1:.3f}"
)

metric_col4.metric(
    "Average Precision",
    f"{avg_precision:.3f}"
)


st.caption(
    "Risk scores are model outputs intended for claim prioritization "
    "and should not be treated as automatic fraud determinations."
)