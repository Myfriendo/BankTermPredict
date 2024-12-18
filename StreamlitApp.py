import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
model_path = 'bank_marketing_pipeline.sav'
model = joblib.load(model_path)

# Streamlit App Title
st.title("Bank Term Deposit Prediction")
st.write("Provide details below to predict if a customer will subscribe to a term deposit.")

# Collect user inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid',
                           'management', 'retired', 'self-employed', 'services',
                           'student', 'technician', 'unemployed', 'unknown'])
marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                       'illiterate', 'professional.course', 'university.degree', 'unknown'])
default = st.selectbox("Default (Has Credit in Default?)", ['yes', 'no', 'unknown'])
housing = st.selectbox("Housing Loan", ['yes', 'no', 'unknown'])
loan = st.selectbox("Personal Loan", ['yes', 'no', 'unknown'])
contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone', 'unknown'])
month = st.selectbox("Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                               'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
day_of_week = st.selectbox("Day of the Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
campaign = st.number_input("Number of Contacts During Campaign", min_value=1, value=1)
pdays = st.number_input("Days Since Last Contact (-1 means client was not contacted)", value=-1)
previous = st.number_input("Number of Previous Contacts", min_value=0, value=0)
emp_var_rate = st.number_input("Employment Variation Rate", value=-1.8)
cons_price_idx = st.number_input("Consumer Price Index", value=93.0)
cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.0)
euribor3m = st.number_input("Euribor 3 Month Rate", value=2.0)
nr_employed = st.number_input("Number of Employees", value=5000.0)

# Organize inputs into a DataFrame
input_data = pd.DataFrame({
    'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
    'default': [default], 'housing': [housing], 'loan': [loan], 'contact': [contact],
    'month': [month], 'day_of_week': [day_of_week], 'campaign': [campaign],
    'pdays': [pdays], 'previous': [previous], 'emp.var.rate': [emp_var_rate],
    'cons.price.idx': [cons_price_idx], 'cons.conf.idx': [cons_conf_idx],
    'euribor3m': [euribor3m], 'nr.employed': [nr_employed]
})

# Predict button
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("The client is likely to subscribe to a term deposit!")
        else:
            st.warning("The client is unlikely to subscribe to a term deposit.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")