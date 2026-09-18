import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/fraud_oracle.csv")

print("Dataset shape:", df.shape)

# -----------------------------
# 2. Target distribution
# -----------------------------
print("\nFraud distribution:")
print(df["FraudFound_P"].value_counts())

print("\nFraud percentage:")
print(df["FraudFound_P"].value_counts(normalize=True) * 100)

# -----------------------------
# 3. Basic statistics
# -----------------------------
print("\nNumerical statistics:")
print(df.describe())

# -----------------------------
# 4. Important categorical values
# -----------------------------
categorical_columns = [
    "AccidentArea",
    "Fault",
    "PolicyType",
    "VehicleCategory",
    "PoliceReportFiled",
    "WitnessPresent",
    "AgentType",
    "BasePolicy"
]

for col in categorical_columns:
    print(f"\n--- {col} ---")
    print(df[col].value_counts())

# -----------------------------
# 5. Fraud by Accident Area
# -----------------------------
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="AccidentArea",
    hue="FraudFound_P"
)

plt.title("Fraud Distribution by Accident Area")
plt.xlabel("Accident Area")
plt.ylabel("Number of Claims")
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Fraud by Fault
# -----------------------------
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Fault",
    hue="FraudFound_P"
)

plt.title("Fraud Distribution by Fault")
plt.xlabel("Fault")
plt.ylabel("Number of Claims")
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Fraud by Base Policy
# -----------------------------
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="BasePolicy",
    hue="FraudFound_P"
)

plt.title("Fraud Distribution by Base Policy")
plt.xlabel("Base Policy")
plt.ylabel("Number of Claims")
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Fraud by Driver Rating
# -----------------------------
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="DriverRating",
    hue="FraudFound_P"
)

plt.title("Fraud Distribution by Driver Rating")
plt.xlabel("Driver Rating")
plt.ylabel("Number of Claims")
plt.tight_layout()
plt.show()