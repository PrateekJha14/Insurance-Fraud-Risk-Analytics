import pandas as pd

# Load dataset
df = pd.read_csv("data/fraud_oracle.csv")

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
for i, column in enumerate(df.columns):
    print(i, ":", column)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())

print("\n========== TARGET DISTRIBUTION ==========")
print(df["FraudFound_P"].value_counts())

print("\n========== TARGET PERCENTAGE ==========")
print(df["FraudFound_P"].value_counts(normalize=True) * 100)