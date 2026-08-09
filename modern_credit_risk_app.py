
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import set_config
import streamlit as st

# ==========================
# Load model and metadata
# ==========================
OUTPUT_DIR = Path("data/modern_credit_risk_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = joblib.load(OUTPUT_DIR / f"model.pkl")
metadata = joblib.load(OUTPUT_DIR / f"metadata.pkl")
PROBA_THRESHOLD = metadata["threshold"]
selected_features = metadata["selected_features"]
categorical_features = metadata["categorical_features"]
continuous_features = metadata["continuous_features"]

# ==========================
# Streamlit App
# ==========================
st.title("Credit Risk Prediction System")
st.write("Enter the Transaction details Below")

col1, col2, col3 = st.columns(3)

with col1:
    age                         = st.number_input("Age", min_value=0, max_value=150, value=35)
    income                      = st.number_input("Annual Income ($)", min_value=0.0, value=55000.0, format="%.2f")
    home_ownership              = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"], index=0)
    employment_length           = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)

with col2:
    purpose                     = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"], index=0)
    grade                       = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"], index=0)
    amount                      = st.number_input("Loan Amount ($)", min_value=0.0, value=15000.0, format="%.2f")
    interest_rate               = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.5, format="%.2f")

with col3:
    percent_income              = st.number_input("Percent of Income (%)", min_value=0.0, max_value=100.0, value=15.0, format="%.2f")
    cb_person_default_on_file   = st.selectbox("Default on File", ["Y", "N"], index=1)
    cb_person_cred_hist_length  = st.number_input("Credit History Length (years)", min_value=0, max_value=100, value=10)

# ==========================
# Build input DataFrame
# ==========================
df = pd.DataFrame([{
    'person_age': age,
    'person_income': income,
    'person_home_ownership': home_ownership,
    'person_emp_length': employment_length,
    'loan_intent': purpose,
    'loan_grade': grade,
    'loan_amnt': amount,
    'loan_int_rate': interest_rate,
    'loan_percent_income': percent_income,
    'cb_person_default_on_file': cb_person_default_on_file,
    'cb_person_cred_hist_length': cb_person_cred_hist_length
}])

# =========================
# Preprocess the input data
# =========================
# Handle missing values
df = df.fillna(-1)

# Income / loan affordability
df["loan_to_income"] = df["loan_amnt"] / (df["person_income"] + 1)
df["income_per_loan_dollar"] = df["person_income"] / (df["loan_amnt"] + 1)

df["monthly_income"] = df["person_income"] / 12
df["monthly_loan_burden"] = df["loan_amnt"] / 12

# Employment stability
df["emp_length_bucket"] = pd.cut(
    df["person_emp_length"],
    bins=[-np.inf, 1, 3, 5, 10, np.inf],
    labels=["0-1", "2-3", "4-5", "6-10", "10+"]
)

# Credit history maturity
df["credit_age_to_person_age"] = (
    df["cb_person_cred_hist_length"] /
    (df["person_age"] + 1)
)
df["credit_history_years"] = df["cb_person_cred_hist_length"]
df["age_at_first_credit"] = (
    df["person_age"] - df["cb_person_cred_hist_length"]
)
df["young_credit_profile"] = (
    df["age_at_first_credit"] < 21
).astype(int)

# Interest-rate features
df["rate_x_loan_amount"] = (
    df["loan_int_rate"] * df["loan_amnt"]
)
df["rate_x_income_ratio"] = (
    df["loan_int_rate"] * df["loan_percent_income"]
)

# Age transformations
df["age_squared"] = df["person_age"] ** 2
df["age_bucket"] = pd.cut(
    df["person_age"],
    bins=[0, 21, 25, 30, 40, 50, 60, np.inf],
    labels=["inf21", "22-25", "26-30", "31-40", "41-50", "51-60", "60+"]
)

# Credit risk interactions
df["loan_burden_x_interest"] = (
    df["loan_percent_income"] * df["loan_int_rate"]
)

# Credit history × employment
df["credit_history_x_emp_length"] = (
    df["cb_person_cred_hist_length"] *
    df["person_emp_length"]
)
df["credit_history_x_age"] = (
    df["cb_person_cred_hist_length"] /
    (df["person_age"] + 1)
)

# =========================
# Predict if the loan is high risk
# =========================
def get_prediction_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

if st.button("Check For Credit Risk"):
    set_config(transform_output="pandas")
    X = df[selected_features].copy()
    proba = get_prediction_scores(model, X)
    result = "High Credit Risk" if proba >= PROBA_THRESHOLD else "Low Credit Risk"
    st.subheader(f"Prediction: {result}")