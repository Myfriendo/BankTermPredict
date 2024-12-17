import streamlit as st
import pickle
import numpy as np

# Define mappings (replace with your actual mappings or dictionaries)
jobs = {"admin": 0, "blue-collar": 1, "entrepreneur": 2, "housemaid": 3, "management": 4, "retired": 5,
        "self-employed": 6, "services": 7, "technician": 8, "unemployed": 9, "unknown": 10}

education_status = {"basic.4y": 0, "basic.6y": 1, "basic.9y": 2, "high.school": 3, "illiterate": 4,
                    "professional.course": 5, "university.degree": 6, "unknown": 7}

marital_status = {"divorced": 0, "married": 1, "single": 2, "unknown": 3}

default_dict = {"no": 0, "yes": 1, "unknown": 2}
housing_dict = {"no": 0, "yes": 1, "unknown": 2}
loan_dict = {"no": 0, "yes": 1, "unknown": 2}

model = pickle.load(open('C:\\Users\\deniz\\PycharmProjects\\BankDepositTermPredict\\medical_insurance_cost_predictor.sav', 'rb'))

# Prediction function
def predict(data):
    array_data = np.asarray(data).reshape(1, -1)
    prediction = model.predict(array_data)
    return prediction

# Streamlit app
st.title("Medical Insurance Cost Prediction")
st.write("Provide the required details below to predict the insurance cost.")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    job = st.selectbox("Job Type", options=jobs.keys())
    marital = st.selectbox("Marital Status", options=marital_status.keys())
    education = st.selectbox("Education Level", options=education_status.keys())
    default = st.selectbox("Has Credit in Default?", options=default_dict.keys())
    housing = st.selectbox("Has Housing Loan?", options=housing_dict.keys())
    loan = st.selectbox("Has Personal Loan?", options=loan_dict.keys())
    balance = st.number_input("Account Balance", value=0.0, step=0.1)

    # Submit button
    submit = st.form_submit_button("Predict")

if submit:
    # Map inputs to numeric values
    input_data = [
        age,
        jobs[job],
        marital_status[marital],
        education_status[education],
        default_dict[default],
        housing_dict[housing],
        loan_dict[loan],
        balance,
    ]

    # Prediction
    result = predict(input_data)

    # Display result
    st.write("### Prediction Result")
    if result == 1:
        st.success("The customer is likely to **subscribe** to medical insurance.")
    else:
        st.warning("The customer is likely to **not subscribe** to medical insurance.")

