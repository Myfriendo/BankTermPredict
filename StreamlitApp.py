import streamlit as st
import pickle
import pandas as pd

# Load the trained pipeline
model_path = 'bank_marketing_pipeline.sav'
with open(model_path, 'rb') as file:
    pipeline = pickle.load(file)

# Streamlit App
st.title("Bank Marketing Campaign Prediction")
st.write("Provide the required details below to predict if a customer will subscribe to the term deposit.")

# Input Form
with st.form("prediction_form"):
    # Numerical Features
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    campaign = st.number_input("Number of Contacts During Campaign", value=1, step=1)
    pdays = st.number_input("Days Since Last Contact (999 = never contacted)", value=999, step=1)
    previous = st.number_input("Number of Contacts Before This Campaign", value=0, step=1)
    emp_var_rate = st.number_input("Employment Variation Rate", value=0.0, step=0.1)
    cons_price_idx = st.number_input("Consumer Price Index", value=93.0, step=0.1)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.0, step=0.1)
    euribor3m = st.number_input("Euribor 3-Month Rate", value=4.0, step=0.1)
    nr_employed = st.number_input("Number of Employees", value=5000.0, step=0.1)

    # Categorical Features
    job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
                               "self-employed", "services", "student", "technician", "unemployed", "unknown"])
    marital = st.selectbox("Marital Status", ["divorced", "married", "single", "unknown"])
    education = st.selectbox("Education Level", ["unknown", "basic.4y", "basic.6y", "basic.9y",
                                                 "high.school", "illiterate", "professional.course", "university.degree"])
    default = st.selectbox("Has Credit in Default?", ["no", "yes", "unknown"])
    housing = st.selectbox("Has Housing Loan?", ["no", "yes", "unknown"])
    loan = st.selectbox("Has Personal Loan?", ["no", "yes", "unknown"])
    contact = st.selectbox("Contact Communication Type", ["cellular", "telephone", "unknown"])
    month = st.selectbox("Month of Last Contact", ["jan", "feb", "mar", "apr", "may", "jun",
                                                   "jul", "aug", "sep", "oct", "nov", "dec"])
    day_of_week = st.selectbox("Day of Week (Last Contact)", ["mon", "tue", "wed", "thu", "fri"])
    poutcome = st.selectbox("Outcome of Previous Campaign", ["unknown", "failure", "other", "success"])

    # Submit Button
    submit = st.form_submit_button("Predict")

if submit:
    # Create DataFrame with expected feature names
    input_data = pd.DataFrame([[
        age, job, marital, education, default, housing, loan, contact, month, day_of_week,
        campaign, pdays, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx,
        euribor3m, nr_employed
    ]], columns=[
        'age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
        'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome',
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
    ])

    # Make a prediction using the loaded pipeline
    prediction = pipeline.predict(input_data)

    # Display Result
    st.write("### Prediction Result")
    if prediction[0] == 1:
        st.success("The customer is likely to **subscribe** to the term deposit.")
    else:
        st.warning("The customer is likely to **not subscribe** to the term deposit.")
