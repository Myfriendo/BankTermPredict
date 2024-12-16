import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Load your model
@st.cache_resource
def load_model():
    with open('medical_insurance_cost_predictor.sav', 'rb') as f:  # Update with your .sav model file
        model = pickle.load(f)
    return model

model = load_model()

# Apply preprocessing
def apply_preprocessing(data):
    label_encoder = LabelEncoder()
    # Example preprocessing; replace with your specific logic
    data['feature1'] = label_encoder.fit_transform(data['feature1'])
    return data

# Streamlit app
def main():
    st.set_page_config(page_title="Prediction App", layout="wide")
    st.title("Prediction App")
    st.subheader("Enter input features and make predictions.")

    # Sidebar information
    st.sidebar.title("App Instructions")
    st.sidebar.markdown("1. Fill in the required features.")
    st.sidebar.markdown("2. Click Predict to see the output.")

    # Input form
    with st.form("input_form"):
        st.write("### Enter Client Data")
        feature1 = st.text_input("Feature 1 (e.g., job):", "unknown")
        feature2 = st.number_input("Feature 2 (e.g., age):", min_value=0, max_value=150, step=1)
        feature3 = st.number_input("Feature 3 (e.g., balance):", value=0.0, step=0.1)
        submitted = st.form_submit_button("Predict")

    # Handle form submission
    if submitted:
        # Simulate data preparation
        input_data = pd.DataFrame({
            'feature1': [feature1],
            'feature2': [feature2],
            'feature3': [feature3]
        })

        preprocessed_data = apply_preprocessing(input_data)

        # Make predictions
        prediction = model.predict(preprocessed_data)
        st.write("### Prediction Result")
        st.write(f"Prediction: {'Yes' if prediction[0] == 1 else 'No'}")

if __name__ == "__main__":
    main()
