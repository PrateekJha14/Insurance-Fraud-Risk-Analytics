# Insurance Fraud Risk Analytics

An end-to-end machine learning application for insurance claim
fraud-risk screening using XGBoost, SHAP explainability, and
Streamlit.

The system assigns a model-generated risk score to insurance claims
and provides claim-level explanations to help prioritize potentially
high-risk claims for further investigation.

---

## Project Overview

Insurance fraud detection is a highly imbalanced classification
problem where fraudulent claims represent a small proportion of
total claims.

This project develops a fraud-risk screening pipeline that:

- Performs exploratory data analysis
- Handles categorical and numerical claim features
- Addresses class imbalance using XGBoost class weighting
- Optimizes the classification threshold using validation data
- Evaluates performance on a held-out test set
- Generates claim-level SHAP explanations
- Provides an interactive Streamlit risk analytics dashboard

---

## Dataset

The dataset contains:

- 15,420 insurance claims
- 33 columns
- 923 fraudulent claims
- Fraud rate: approximately 5.99%

The target variable is:

`FraudFound_P`

The dataset is not included in this repository.

---

## Machine Learning Pipeline

```text
Insurance Claims
       |
       v
Data Preprocessing
       |
       +--> Numerical Features
       |       |
       |       +--> Median Imputation
       |       +--> Standard Scaling
       |
       +--> Categorical Features
               |
               +--> Most-Frequent Imputation
               +--> One-Hot Encoding
       |
       v
XGBoost Classifier
       |
       v
Fraud Risk Score
       |
       v
Validation-Based Threshold Selection
       |
       +--------------------+
       |                    |
       v                    v
Risk Classification       SHAP
                           |
                           v
                    Claim Explanation
       |
       v
Streamlit Dashboard

Model Performance

The final XGBoost model was evaluated on a held-out test set.

Metric	Score
ROC-AUC	0.8383
Average Precision	0.2548
Precision	0.1816
Recall	0.7027
F1 Score	0.2886

The test-set confusion matrix was:

[[2313  586]
 [  55  130]]

Because fraud represents only approximately 6% of claims, accuracy
alone is not an appropriate measure of model effectiveness. The
project therefore focuses on recall, precision, F1, ROC-AUC and
Average Precision.

Class Imbalance

The training data contained substantially more legitimate claims
than fraudulent claims.

The XGBoost model uses:

scale_pos_weight

to give additional importance to the minority fraud class during
training.

Threshold Optimization

Rather than relying only on the default classification threshold,
candidate thresholds were evaluated on the validation set using F1
score.

The selected validation threshold was:

0.49

This threshold was then applied to the held-out test set.

Explainability

SHAP (SHapley Additive exPlanations) is used to understand model
behavior.

Two levels of explainability are provided:

Global Explainability

Feature importance and SHAP summary analysis identify features that
strongly influence model predictions across claims.

Local Explainability

For an individual claim, SHAP values show which encoded claim
characteristics pushed the prediction toward or away from the fraud
class.

Example:

BasePolicy_Liability        +0.410
Fault_Policy Holder         +0.338
NumberOfSuppliments_3 to 5  -0.302
MonthClaimed_Feb            -0.178

SHAP values explain model behavior and should not be interpreted as
causal relationships.

Streamlit Dashboard

The application provides:

Portfolio Overview
Total claims
Fraud rate
Risk distribution
Fraud distribution
Risk-score distribution
Feature importance
Model performance
Claim Investigation

Users can select individual claims and view:

Fraud risk score
Risk level
Claim characteristics
Model classification
SHAP feature contributions
Risk interpretation

The dashboard is intended for investigation prioritization and does
not automatically establish that a claim is fraudulent.

Tech Stack
Python
Pandas
NumPy
Scikit-learn
XGBoost
SHAP
Matplotlib
Streamlit
Joblib
Project Structure
Insurance-Fraud-Risk-Analytics/
|
├── data/
│   └── README.md
|
├── models/
│   ├── feature_importance.png
│   └── shap_summary.png
|
├── src/
│   ├── eda.py
│   ├── train_model.py
│   ├── threshold_analysis.py
│   ├── final_model.py
│   ├── explainability.py
│   └── test_shap.py
|
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
Running the Project
1. Clone the repository
git clone <repository-url>
cd Insurance-Fraud-Risk-Analytics
2. Create a virtual environment
python -m venv venv
3. Activate it

Windows:

venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
5. Add the dataset

Place the dataset at:

data/fraud_oracle.csv
6. Train the model
python src/final_model.py
7. Generate explainability outputs
python src/explainability.py
8. Launch the dashboard
streamlit run app.py