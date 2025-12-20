import streamlit as st
import requests
import os

# API_URL = os.environ.get("API_URL", "http://localhost:5000/predict")
API_URL = os.environ.get("API_URL")

st.title("Credit Risk Prediction")
st.write("Predict probability of loan default")

form = st.form("credit_form")

inputs = {
    "status": form.selectbox("Status", ["A11", "A12", "A13", "A14"]),
    "duration": form.slider("Duration (months)", 4, 72, 12),
    "credit_history": form.selectbox("Credit History", ["A30", "A31", "A32", "A33", "A34"]),
    "purpose": form.selectbox("Purpose", ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"]),
    "credit_amount": form.number_input("Credit Amount", 250, 20000, 2000),
    "savings": form.selectbox("Savings", ["A61", "A62", "A63", "A64", "A65"]),
    "employment": form.selectbox("Employment", ["A71", "A72", "A73", "A74", "A75"]),
    "installment_rate": form.slider("Installment Rate", 1, 4, 2),
    "personal_status_sex": form.selectbox("Personal Status", ["A91", "A92", "A93", "A94", "A95"]),
    "other_debtors": form.selectbox("Other Debtors", ["A101", "A102", "A103"]),
    "residence_since": form.slider("Residence Since", 1, 4, 2),
    "property": form.selectbox("Property", ["A121", "A122", "A123", "A124"]),
    "age": form.slider("Age", 18, 75, 35),
    "other_installment_plans": form.selectbox("Other Installment Plans", ["A141", "A142", "A143"]),
    "housing": form.selectbox("Housing", ["A151", "A152", "A153"]),
    "existing_credits": form.slider("Existing Credits", 1, 4, 1),
    "job": form.selectbox("Job", ["A171", "A172", "A173", "A174"]),
    "num_dependents": form.slider("Dependents", 1, 2, 1),
    "own_telephone": form.selectbox("Telephone", ["A191", "A192"]),
    "foreign_worker": form.selectbox("Foreign Worker", ["A201", "A202"]),
}

submitted = form.form_submit_button("Predict")

if submitted:
    response = requests.post(API_URL, json=inputs)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Default probability: {result['default_probability']:.2%}")
    else:
        st.error(response.text)
