# 🛡️ Insurance Fraud Risk Analytics

> An end-to-end machine learning application for **insurance claim fraud-risk screening** using XGBoost, class-imbalance handling, threshold optimization, SHAP explainability, and an interactive Streamlit analytics dashboard.

The system assigns a **model-generated fraud risk score** to insurance claims and provides claim-level explanations to help prioritize potentially high-risk claims for further investigation.

---

## 📌 Overview

Insurance fraud detection is a highly imbalanced classification problem where fraudulent claims represent only a small proportion of total claims.

This project develops an end-to-end fraud-risk screening pipeline that:

* Performs exploratory data analysis
* Handles numerical and categorical claim features
* Applies preprocessing using `ColumnTransformer`
* Addresses class imbalance using XGBoost's `scale_pos_weight`
* Evaluates multiple baseline models
* Trains a final XGBoost classifier
* Optimizes the classification threshold using validation data
* Evaluates the final model on a held-out test set
* Generates global feature-importance analysis
* Uses SHAP for model explainability
* Provides an interactive Streamlit dashboard for claim investigation

The goal is **not to automatically declare a claim fraudulent**, but to generate risk scores that can help prioritize claims for further investigation.

---

# 🎯 Problem Statement

Insurance fraud creates financial and operational challenges for insurers. Because fraudulent claims are much less frequent than legitimate claims, a model that simply maximizes overall accuracy may fail to identify enough fraudulent cases.

This project therefore focuses on metrics such as:

* Precision
* Recall
* F1 Score
* ROC-AUC
* Average Precision

rather than relying on accuracy alone.

The model produces a probability-like **fraud risk score**, which is then converted into a risk category for dashboard-based investigation.

---

# 📊 Dataset

The project uses an insurance claims dataset containing:

| Property          |      Value |
| ----------------- | ---------: |
| Total claims      | **15,420** |
| Columns           |     **33** |
| Fraudulent claims |    **923** |
| Fraud rate        | **~5.99%** |

The target variable is:

```text
FraudFound_P
```

where:

```text
0 → Non-fraudulent claim
1 → Fraudulent claim
```

The dataset is **not included in the repository**. It must be placed at:

```text
data/fraud_oracle.csv
```

The project removes `PolicyNumber` before model training because it is treated as an identifier rather than a predictive feature.

---

# 🔄 Machine Learning Pipeline

```text
                  Insurance Claims
                         │
                         ▼
                  Data Exploration
                         │
                         ▼
               Remove PolicyNumber
                         │
                         ▼
                 Feature / Target
                    Separation
                         │
                         ▼
              Train / Validation / Test
                         │
                         ▼
              ┌──────────────────────┐
              │   Preprocessing      │
              ├──────────────────────┤
              │ Numerical Features   │
              │ • Median Imputation  │
              │ • Standard Scaling   │
              │                      │
              │ Categorical Features │
              │ • Mode Imputation    │
              │ • One-Hot Encoding   │
              └──────────┬───────────┘
                         │
                         ▼
                  XGBoost Classifier
                         │
                         ▼
                Fraud Probability
                         │
                         ▼
              Validation Threshold
                  Optimization
                         │
                         ▼
                Fraud Risk Score
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Risk Classification         SHAP
             │                       │
             ▼                       ▼
       Streamlit Dashboard    Claim Explanation
```

---

# 🧹 Data Preprocessing

The preprocessing pipeline is implemented using Scikit-learn's `Pipeline` and `ColumnTransformer`.

## Numerical Features

Numerical columns are processed using:

```text
Missing values
      ↓
Median Imputation
      ↓
Standard Scaling
```

Implemented using:

```python
SimpleImputer(strategy="median")
StandardScaler()
```

---

## Categorical Features

Categorical columns are processed using:

```text
Missing values
      ↓
Most-Frequent Imputation
      ↓
One-Hot Encoding
```

Implemented using:

```python
SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")
```

Using `handle_unknown="ignore"` allows unseen categorical values to be handled during inference without breaking the pipeline.

---

# ⚖️ Handling Class Imbalance

Fraudulent claims represent only approximately **6%** of the dataset.

To give the minority fraud class greater importance during training, the final XGBoost model uses:

```python
scale_pos_weight = negative_samples / positive_samples
```

The calculated weight is passed to:

```python
XGBClassifier(
    scale_pos_weight=scale_pos_weight
)
```

This helps the model pay more attention to fraudulent claims during training.

---

# 🤖 Model Development

The project includes an initial model comparison using:

* Logistic Regression
* Random Forest
* XGBoost

This comparison is implemented in:

```text
src/train_model.py
```

The final model is trained separately using the more refined pipeline in:

```text
src/final_model.py
```

The final classifier is:

```text
XGBoost
```

with:

```python
n_estimators = 400
max_depth = 4
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
```

along with the calculated `scale_pos_weight`.

---

# ✂️ Train / Validation / Test Strategy

The final pipeline uses a two-stage split.

### Stage 1

```text
80% → Training + Validation
20% → Held-out Test
```

### Stage 2

The 80% training portion is further divided:

```text
75% → Training
25% → Validation
```

This produces approximately:

```text
Training    → 9,252 claims
Validation  → 3,084 claims
Test        → 3,084 claims
```

The validation set is used for **threshold optimization**, while the test set remains held out for final evaluation.

---

# 🎚️ Threshold Optimization

Instead of automatically using the default classification threshold of `0.50`, the project evaluates candidate probability thresholds on the validation set.

Thresholds from:

```text
0.05 → 0.50
```

are evaluated in increments of:

```text
0.01
```

For each threshold, the validation F1 score is calculated.

The threshold producing the highest validation F1 score is selected.

### Selected threshold

```text
0.49
```

This threshold is then applied **once to the held-out test predictions**.

This approach separates:

```text
Model training
      ↓
Threshold selection
      ↓
Final test evaluation
```

rather than selecting the threshold directly on the test set.

---

# 📈 Final Model Performance

The final XGBoost model was evaluated on **3,084 held-out test claims**.

| Metric                   |      Score |
| ------------------------ | ---------: |
| ROC-AUC                  | **0.8383** |
| Average Precision        | **0.2548** |
| Precision                | **0.1816** |
| Recall                   | **0.7027** |
| F1 Score                 | **0.2886** |
| Classification Threshold |   **0.49** |

These values are calculated from the saved test predictions generated by the final model pipeline.

---

## 🧮 Confusion Matrix

At the selected threshold of `0.49`:

```text
                Predicted
              Non-Fraud  Fraud

Actual
Non-Fraud       2313      586

Fraud             55      130
```

Therefore:

* **2,313** legitimate claims were correctly classified as non-fraud
* **586** legitimate claims were flagged for review
* **130** fraudulent claims were detected
* **55** fraudulent claims were missed

The resulting fraud recall is approximately:

```text
130 / (130 + 55) = 70.27%
```

---

# 📊 Why Accuracy Is Not the Main Metric

With only around 6% of claims being fraudulent, a model could achieve high accuracy simply by predicting most claims as legitimate.

For fraud screening, missing fraudulent claims can be important, so this project reports:

```text
Precision
Recall
F1
ROC-AUC
Average Precision
```

The project therefore emphasizes the trade-off between identifying fraudulent claims and generating false investigation flags.

---

# 🧠 Explainability with SHAP

The project uses **SHAP (SHapley Additive exPlanations)** to understand model predictions.

SHAP is used at two levels:

### Global Explainability

The project calculates XGBoost feature importance and generates a SHAP summary analysis across a sample of test claims.

Outputs include:

```text
models/feature_importance.csv
models/feature_importance.png
models/shap_summary.png
```

The SHAP analysis uses up to **500 sampled test claims** to keep the computation manageable.

---

### Local Explainability

For an individual claim, SHAP values show how individual encoded features influenced the model's prediction.

```text
Positive SHAP value
        ↓
Pushes prediction toward fraud

Negative SHAP value
        ↓
Pushes prediction away from fraud
```

For example, the project can display features such as:

```text
BasePolicy_Liability
Fault_Policy Holder
NumberOfSuppliments_3 to 5
MonthClaimed_Feb
```

with their corresponding SHAP contributions.

> **Important:** SHAP values explain model behavior; they should not be interpreted as causal relationships.

---

# 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard with two main sections.

---

## 1. 📊 Portfolio Overview

The Portfolio Overview page provides a high-level view of the test/claim portfolio.

### Key metrics

* Total claims
* Fraud claims
* Fraud rate
* ROC-AUC
* Precision
* Recall
* F1 Score
* Average Precision

### Visualizations

The dashboard displays:

* Claim risk distribution
* Actual fraud distribution
* Fraud risk score distribution
* Top model features
* Model performance

Users can also filter claims by:

```text
Low Risk
Medium Risk
High Risk
```

---

# 🚨 Risk Categorization

The model's fraud probability is converted into a score from `0–100`:

```text
Risk Score = Fraud Probability × 100
```

The dashboard then categorizes claims as:

| Risk Score | Risk Level  |
| ---------: | ----------- |
|     `< 30` | Low Risk    |
| `30 – <70` | Medium Risk |
|     `≥ 70` | High Risk   |

These categories are **dashboard screening bands**, not independent fraud labels.

---

# 🔎 2. Claim Investigation

The Claim Investigation page allows the user to select an individual claim from the test predictions.

For each claim, the dashboard displays:

### Risk Assessment

* Fraud Risk Score
* Risk Level
* Model Classification

The model classification is displayed as:

```text
Review
```

or:

```text
No Review Flag
```

based on the model's selected classification threshold.

---

### Claim Information

The dashboard displays claim characteristics such as:

* Month
* Accident Area
* Fault
* Policy Type
* Base Policy
* Vehicle Category
* Vehicle Price
* Age
* Driver Rating
* Deductible
* Police Report Filed
* Witness Present
* Agent Type
* Past Number of Claims
* Number of Supplements
* Address Change
* Number of Cars
* Age of Vehicle
* Age of Policy Holder

---

### 💡 Individual Claim Explanation

For the selected claim, the application:

```text
Selected Claim
      ↓
Preprocessing Pipeline
      ↓
XGBoost Model
      ↓
SHAP TreeExplainer
      ↓
Top 10 Feature Contributions
```

The dashboard displays:

* Feature
* SHAP Value
* Direction of influence

For example:

```text
Positive SHAP → Toward Fraud
Negative SHAP → Away from Fraud
```

This provides an interpretable explanation of the model's risk score.

---

# 🗂️ Project Structure

```text
Insurance-Fraud-Risk-Analytics/
│
├── data/
│   └── fraud_oracle.csv          # Dataset (not included in repo)
│
├── models/
│   ├── fraud_model.pkl           # Trained preprocessing + XGBoost pipeline
│   ├── test_predictions.csv      # Held-out test predictions
│   ├── feature_importance.csv    # Global XGBoost feature importance
│   ├── feature_importance.png    # Feature importance visualization
│   └── shap_summary.png          # SHAP summary visualization
│
├── src/
│   ├── eda.py                    # Exploratory data analysis
│   ├── train_model.py            # Baseline model comparison
│   ├── threshold_analysis.py     # Probability threshold analysis
│   ├── final_model.py            # Final XGBoost training pipeline
│   ├── explainability.py         # Feature importance + SHAP analysis
│   └── test_shap.py              # Individual claim SHAP verification
│
├── app.py                        # Streamlit dashboard
├── app_backup.py                 # Backup dashboard implementation
├── inspectdata.py                # Dataset inspection utility
├── requirements.txt
├── .gitignore
└── README.md
```

The repository structure and primary workflow are also reflected in the current GitHub repository.

---

# 🛠️ Tech Stack

| Technology   | Purpose                                 |
| ------------ | --------------------------------------- |
| Python       | Core development                        |
| Pandas       | Data manipulation                       |
| NumPy        | Numerical computation                   |
| Scikit-learn | Preprocessing, splitting and evaluation |
| XGBoost      | Final fraud-risk classifier             |
| SHAP         | Model explainability                    |
| Matplotlib   | Visualization                           |
| Seaborn      | Exploratory data analysis               |
| Streamlit    | Interactive dashboard                   |
| Joblib       | Model serialization                     |

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/PrateekJha14/Insurance-Fraud-Risk-Analytics.git
cd Insurance-Fraud-Risk-Analytics
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📂 Dataset Setup

The dataset is not included in the repository.

Place the dataset at:

```text
data/fraud_oracle.csv
```

The expected target column is:

```text
FraudFound_P
```

The training scripts expect the original dataset structure used by the project.

---

# ▶️ Train the Model

To train the final XGBoost pipeline:

```bash
python src/final_model.py
```

This performs:

1. Dataset loading
2. Identifier removal
3. Feature/target separation
4. Train/validation/test splitting
5. Class imbalance calculation
6. Numerical preprocessing
7. Categorical preprocessing
8. XGBoost training
9. Validation threshold optimization
10. Held-out test evaluation
11. Model serialization
12. Test prediction export

The trained pipeline is saved to:

```text
models/fraud_model.pkl
```

Test predictions are saved to:

```text
models/test_predictions.csv
```

---

# 🧠 Generate Explainability Outputs

Run:

```bash
python src/explainability.py
```

This generates:

```text
models/feature_importance.csv
models/feature_importance.png
models/shap_summary.png
```

---

# 🔬 Run Threshold Analysis

To independently analyze different classification thresholds:

```bash
python src/threshold_analysis.py
```

The script evaluates precision, recall and F1 across different probability thresholds and reports the threshold with the highest F1 score for that analysis.

---

# 📊 Run Exploratory Data Analysis

Run:

```bash
python src/eda.py
```

The EDA script examines:

* Dataset shape
* Fraud distribution
* Numerical statistics
* Categorical distributions
* Fraud by accident area
* Fraud by fault
* Fraud by base policy
* Fraud by driver rating

---

# 🔍 Test Individual SHAP Explanation

The repository also includes:

```bash
python src/test_shap.py
```

This verifies the SHAP explanation for an individual claim and compares the model's current probability with the saved prediction.

---

# 🚀 Launch the Dashboard

After the trained model and prediction artifacts are available:

```bash
streamlit run app.py
```

The dashboard loads:

```text
models/test_predictions.csv
models/feature_importance.csv
models/fraud_model.pkl
data/fraud_oracle.csv
```

and provides the Portfolio Overview and Claim Investigation interfaces.

---

# 📌 Key Design Decisions

## 1. Probability-based risk scoring

Rather than treating the model output as a simple binary label, the system preserves the predicted fraud probability and converts it into a risk score:

```text
Fraud Probability × 100
```

This provides a more granular measure for prioritization.

---

## 2. Validation-based threshold selection

The classification threshold is selected using a validation set rather than tuning directly on the test set.

This helps keep the final test evaluation separate from threshold selection.

---

## 3. Class imbalance handling

Because fraudulent claims are a minority class, XGBoost uses:

```text
scale_pos_weight
```

to increase the influence of fraud examples during training.

---

## 4. Explainability

The system does not stop at a risk score.

It provides:

```text
Risk Score
     +
Risk Category
     +
Global Feature Importance
     +
Local SHAP Explanation
```

This allows an investigator to inspect which encoded claim characteristics contributed most strongly to an individual model prediction.

---

# 💼 Business Use Case

A potential operational workflow is:

```text
                    Incoming Claims
                           │
                           ▼
                    Fraud Risk Model
                           │
                           ▼
                    Fraud Probability
                           │
                           ▼
                    Risk Categorization
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Low Risk     Medium Risk     High Risk
             │             │             │
             ▼             ▼             ▼
        Lower review    Additional     Prioritize
        priority        review         investigation
                           │
                           ▼
                    Human Investigation
```

The model is therefore positioned as a **screening and prioritization system**, rather than an autonomous fraud-decision system.

The Streamlit application explicitly states that risk scores are intended for claim prioritization and should not be treated as automatic fraud determinations.

---

# ⚠️ Limitations

This project is a machine-learning **risk-screening prototype** and has several limitations:

* The dataset represents historical claims and may not reflect current fraud patterns.
* Fraudulent claims are relatively rare, creating substantial class imbalance.
* The model can generate false positives.
* Some fraudulent claims can remain undetected.
* Risk categories are based on fixed score bands.
* SHAP explanations describe model behavior rather than causal relationships.
* The model should not be interpreted as proof that a claim is fraudulent.
* Real-world deployment would require additional validation, monitoring, governance, and human review.

---

# 🔮 Future Improvements

Potential extensions include:

* [ ] Hyperparameter optimization with cross-validation
* [ ] Calibration of predicted probabilities
* [ ] Cost-sensitive threshold optimization based on investigation capacity
* [ ] Precision-Recall curve visualization
* [ ] ROC curve visualization
* [ ] Model monitoring for data drift
* [ ] Probability calibration monitoring
* [ ] Automated investigation queues
* [ ] Batch scoring for newly submitted claims
* [ ] Integration with claims-management systems
* [ ] Human-in-the-loop investigation workflow
* [ ] Additional fraud-specific feature engineering
* [ ] More extensive cross-validation and temporal validation
* [ ] Model versioning and experiment tracking

---

# 📈 Key Takeaways

The project demonstrates an end-to-end machine-learning workflow for imbalanced insurance fraud screening:

```text
EDA
 ↓
Preprocessing
 ↓
Baseline Model Comparison
 ↓
XGBoost
 ↓
Class Imbalance Handling
 ↓
Validation Threshold Optimization
 ↓
Held-out Test Evaluation
 ↓
SHAP Explainability
 ↓
Streamlit Risk Analytics
```

The final model achieved:

```text
ROC-AUC          0.8383
Average Precision 0.2548
Recall           0.7027
F1 Score         0.2886
```

at the validation-selected threshold of:

```text
0.49
```

The resulting application turns those model outputs into an interactive **claim-risk screening and investigation interface**.

---

# 👨‍💻 Author

**Prateek Jha**

B.Tech — Chemical Engineering
Indian Institute of Technology Patna

GitHub: [@PrateekJha14](https://github.com/PrateekJha14)

---

## 📄 License

This project is intended for educational and research purposes.
