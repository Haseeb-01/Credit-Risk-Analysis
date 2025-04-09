import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("random_forest_model.pkl")  # Or xgb_model.pkl
scaler = joblib.load("scaler.pkl")  # Load the scaler used during training

# Set page configuration
st.set_page_config(page_title="Credit Risk Prediction", layout="centered")
st.title("💳 Credit Risk Prediction App")
st.markdown("Enter customer details below to assess credit risk.")

# Create two columns for parallel input fields
col1, col2 = st.columns(2)

# Input fields split across two columns (5 in each)
with col1:
    RevolvingUtilizationOfUnsecuredLines = st.number_input("Revolving Utilization (%)", min_value=0.0, value=0.0)
    age = st.number_input("Age", min_value=18, step=1, value=18)
    NumberOfTime30_59DaysPastDueNotWorse = st.number_input("LatePayment_30_59_Days", min_value=0, step=1, value=0)
    DebtRatio = st.number_input("Debt Ratio", min_value=0.0, value=0.0)
    MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=0.0)

with col2:
    NumberOfOpenCreditLinesAndLoans = st.number_input("Open Credit Lines & Loans", min_value=0, step=1, value=0)
    NumberOfTimes90DaysLate = st.number_input("Number of Times 90 Days Late", min_value=0, step=1, value=0)
    NumberRealEstateLoansOrLines = st.number_input("Number of Real Estate Loans", min_value=0, step=1, value=0)
    NumberOfTime60_89DaysPastDueNotWorse = st.number_input("Number of Times 60-89 Days Late (Not Worse)", min_value=0, step=1, value=0)
    NumberOfDependents = st.number_input("Number of Dependents", min_value=0, step=1, value=0)

# Prepare input data in the correct order
input_data = np.array([[RevolvingUtilizationOfUnsecuredLines, age, NumberOfTime30_59DaysPastDueNotWorse,
                        DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans,
                        NumberOfTimes90DaysLate, NumberRealEstateLoansOrLines,
                        NumberOfTime60_89DaysPastDueNotWorse, NumberOfDependents]])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict Credit Risk"):
    # Predict
    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]  # Prob of default

    # Show results
    if prediction == 1:
        st.error(f"⚠️ High Risk Customer (Default Probability: {probability:.2%})")
        st.write("❌ Recommendation: Do not approve credit or increase terms.")
    else:
        st.success(f"✅ Low Risk Customer (Default Probability: {probability:.2%})")
        st.write("✔️ Recommendation: Safe to approve.")