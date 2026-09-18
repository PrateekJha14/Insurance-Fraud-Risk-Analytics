import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
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


@st.cache_resource
def load_model():

    return joblib.load(
        "models/fraud_model.pkl"
    )


predictions, importance, original_data = load_data()
model = load_model()

preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["model"]

explainer = shap.TreeExplainer(xgb_model)


# ============================================================
# PREPARE RISK DATA
# ============================================================

predictions["Risk Score"] = (
    predictions["Fraud_Probability"] * 100
)


def risk_category(score):

    if score >= 70:
        return "High Risk"

    elif score >= 30:
        return "Medium Risk"

    return "Low Risk"


predictions["Risk Level"] = (
    predictions["Risk Score"]
    .apply(risk_category)
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ Risk Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Portfolio Overview",
        "Claim Investigation"
    ]
)


# ============================================================
# PORTFOLIO OVERVIEW
# ============================================================

if page == "Portfolio Overview":

    st.title(
        "🛡️ Insurance Fraud Risk Analytics"
    )

    st.markdown(
        """
        **Machine-learning powered insurance claim risk screening**

        XGBoost is used to assign risk scores to insurance claims
        and prioritize potentially high-risk claims for further
        investigation.
        """
    )

    st.divider()


    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    total_claims = len(
        original_data
    )

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
        predictions["Fraud_Probability"]
        >= threshold
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


    # --------------------------------------------------------
    # RISK FILTER
    # --------------------------------------------------------

    st.sidebar.subheader(
        "Risk Filters"
    )

    risk_filter = st.sidebar.multiselect(
        "Risk Level",
        [
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


    filtered_predictions = predictions[
        predictions["Risk Level"].isin(
            risk_filter
        )
    ]


    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Claim Risk Distribution"
        )

        risk_counts = (
            filtered_predictions[
                "Risk Level"
            ]
            .value_counts()
            .reindex(
                [
                    "Low Risk",
                    "Medium Risk",
                    "High Risk"
                ],
                fill_value=0
            )
        )

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.bar(
            risk_counts.index,
            risk_counts.values
        )

        ax.set_xlabel(
            "Risk Level"
        )

        ax.set_ylabel(
            "Number of Claims"
        )

        ax.set_title(
            "Claims by Risk Level"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    with col2:

        st.subheader(
            "Actual Fraud Distribution"
        )

        fraud_counts = (
            original_data[
                "FraudFound_P"
            ]
            .value_counts()
        )

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.bar(
            [
                "Legitimate",
                "Fraud"
            ],
            [
                fraud_counts.get(
                    0,
                    0
                ),
                fraud_counts.get(
                    1,
                    0
                )
            ]
        )

        ax.set_ylabel(
            "Number of Claims"
        )

        ax.set_title(
            "Actual Claim Distribution"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # --------------------------------------------------------
    # RISK SCORE
    # --------------------------------------------------------

    st.subheader(
        "Fraud Risk Score Distribution"
    )

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
        label=(
            f"Model Threshold "
            f"({threshold:.2f})"
        )
    )

    ax.set_xlabel(
        "Fraud Risk Score"
    )

    ax.set_ylabel(
        "Number of Claims"
    )

    ax.set_title(
        "Distribution of Model Risk Scores"
    )

    ax.legend()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "Top Features Influencing Fraud Risk"
    )

    top_features = (
        importance
        .head(12)
        .copy()
    )

    top_features["Feature"] = (
        top_features["Feature"]
        .str.replace(
            "cat__",
            "",
            regex=False
        )
        .str.replace(
            "num__",
            "",
            regex=False
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    ax.set_xlabel(
        "Model Feature Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Model Performance"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Precision",
        f"{precision:.3f}"
    )

    m2.metric(
        "Recall",
        f"{recall:.3f}"
    )

    m3.metric(
        "F1 Score",
        f"{f1:.3f}"
    )

    m4.metric(
        "Average Precision",
        f"{avg_precision:.3f}"
    )


    st.caption(
        """
        Risk scores are model outputs intended for claim
        prioritization and should not be treated as automatic
        fraud determinations.
        """
    )


# ============================================================
# CLAIM INVESTIGATION
# ============================================================

else:

    st.title(
        "🔎 Claim Investigation"
    )

    st.markdown(
        """
        Select a claim from the test set to inspect its
        model-generated fraud risk score, claim characteristics,
        and feature-level explanation.
        """
    )

    st.divider()


    # ========================================================
    # CLAIM SELECTION
    # ========================================================

    claim_indices = predictions.index.tolist()

    selected_claim = st.selectbox(
        "Select Claim",
        claim_indices
    )

    claim_row = predictions.iloc[
        selected_claim
    ]


    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

    risk_score = (
        claim_row["Fraud_Probability"] * 100
    )

    risk_level = (
        "High Risk"
        if risk_score >= 70
        else
        "Medium Risk"
        if risk_score >= 30
        else
        "Low Risk"
    )

    predicted_fraud = int(
        claim_row["Predicted_Fraud"]
    )


    st.subheader(
        "Risk Assessment"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Fraud Risk Score",
        f"{risk_score:.1f}%"
    )

    c2.metric(
        "Risk Level",
        risk_level
    )

    c3.metric(
        "Model Classification",
        (
            "Review"
            if predicted_fraud == 1
            else
            "No Review Flag"
        )
    )


    st.divider()


    # ========================================================
    # CLAIM INFORMATION
    # ========================================================

    st.subheader(
        "Claim Information"
    )


    details = {
        "Month":
            claim_row["Month"],

        "Accident Area":
            claim_row["AccidentArea"],

        "Fault":
            claim_row["Fault"],

        "Policy Type":
            claim_row["PolicyType"],

        "Base Policy":
            claim_row["BasePolicy"],

        "Vehicle Category":
            claim_row["VehicleCategory"],

        "Vehicle Price":
            claim_row["VehiclePrice"],

        "Age":
            claim_row["Age"],

        "Driver Rating":
            claim_row["DriverRating"],

        "Deductible":
            claim_row["Deductible"],

        "Police Report Filed":
            claim_row["PoliceReportFiled"],

        "Witness Present":
            claim_row["WitnessPresent"],

        "Agent Type":
            claim_row["AgentType"],

        "Past Number of Claims":
            claim_row["PastNumberOfClaims"],

        "Number of Supplements":
            claim_row["NumberOfSuppliments"],

        "Address Change":
            claim_row["AddressChange_Claim"],

        "Number of Cars":
            claim_row["NumberOfCars"],

        "Age of Vehicle":
            claim_row["AgeOfVehicle"],

        "Age of Policy Holder":
            claim_row["AgeOfPolicyHolder"]
    }


    details_df = pd.DataFrame(
        list(details.items()),
        columns=[
            "Attribute",
            "Value"
        ]
    )


    st.dataframe(
        details_df,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.subheader(
        "💡 Why did the model give this risk score?"
    )

    st.markdown(
        """
        SHAP values show how individual claim characteristics
        influenced the model's prediction.

        **Positive SHAP values** push the prediction toward the
        fraud class.

        **Negative SHAP values** push the prediction away from
        the fraud class.
        """
    )


    # --------------------------------------------------------
    # PREPARE CLAIM FOR SHAP
    # --------------------------------------------------------

    claim_for_model = (
        claim_row
        .drop(
            labels=[
                "Actual_Fraud",
                "Fraud_Probability",
                "Predicted_Fraud"
            ]
        )
        .to_frame()
        .T
    )


    # --------------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------------

    transformed_claim = (
        preprocessor.transform(
            claim_for_model
        )
    )


    # --------------------------------------------------------
    # SHAP VALUES
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        transformed_claim
    )[0]


    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values
    })


    shap_df["Absolute Impact"] = (
        shap_df["SHAP Value"]
        .abs()
    )


    # Top 10 by absolute impact

    top_shap = (
        shap_df
        .sort_values(
            "Absolute Impact",
            ascending=False
        )
        .head(10)
        .copy()
    )


    # Clean feature names

    top_shap["Feature"] = (
        top_shap["Feature"]
        .str.replace(
            "cat__",
            "",
            regex=False
        )
        .str.replace(
            "num__",
            "",
            regex=False
        )
    )


    # ========================================================
    # SHAP TABLE
    # ========================================================

    display_shap = top_shap[
        [
            "Feature",
            "SHAP Value"
        ]
    ].copy()


    display_shap["Direction"] = (
        display_shap["SHAP Value"]
        .apply(
            lambda x:
                "↑ Toward Fraud"
                if x > 0
                else
                "↓ Away from Fraud"
        )
    )


    display_shap["SHAP Value"] = (
        display_shap["SHAP Value"]
        .round(3)
    )


    st.dataframe(
        display_shap,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # SHAP BAR CHART
    # ========================================================

    st.subheader(
        "Feature Impact"
    )


    plot_data = (
        top_shap
        .sort_values(
            "SHAP Value"
        )
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    ax.barh(
        plot_data["Feature"],
        plot_data["SHAP Value"]
    )


    ax.axvline(
        0,
        linewidth=1
    )


    ax.set_xlabel(
        "SHAP Value"
    )


    ax.set_ylabel(
        "Feature"
    )


    ax.set_title(
        "Features Influencing This Claim's Risk Score"
    )


    st.pyplot(
        fig,
        use_container_width=True
    )


    plt.close(fig)


    st.divider()


    # ========================================================
    # RISK SCORE
    # ========================================================

    st.subheader(
        "Risk Score"
    )


    st.progress(
        min(
            int(risk_score),
            100
        )
    )


    st.caption(
        f"""
        Model risk score: {risk_score:.1f}/100.
        Higher scores indicate stronger model association
        with the fraud class in this dataset.
        """
    )


    # ========================================================
    # INVESTIGATION GUIDANCE
    # ========================================================

    if risk_level == "High Risk":

        st.warning(
            """
            This claim falls into the high-risk band based on
            the model's risk score and can be prioritized for
            further investigation.
            """
        )

    elif risk_level == "Medium Risk":

        st.info(
            """
            This claim falls into the medium-risk band and may
            warrant additional review depending on investigation
            capacity and business rules.
            """
        )

    else:

        st.success(
            """
            This claim falls into the low-risk band according
            to the model's current risk categorization.
            """
        )


    st.caption(
        """
        This tool supports investigation prioritization.
        It does not establish that a claim is fraudulent.
        """
    )