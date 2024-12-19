# Bank Term Prediction Streamlit App

## Overview
This repository contains a Streamlit application for predicting whether a customer will subscribe to a term deposit based on their attributes. The app leverages a machine learning model stored in the file `bank_marketing_pipeline.sav`.

## Prerequisites
1. **Python Installation:** Ensure Python 3.7 or higher is installed on your system.
   - You can download it from [python.org](https://www.python.org/).

2. **Required Libraries:** Install the following Python libraries:
   - Streamlit
   - pandas
   - joblib

   Run the following command in your terminal or command prompt:
   ```
   pip install streamlit pandas joblib
   ```

3. **Model File:**
   - Download the `bank_marketing_pipeline.sav` file from this repository.
   - Place the file in the same directory as the `StreamlitApp.py` script.

4. **Clone the Repository (Optional):**
   - If you want to clone the entire repository, use the following command:
     ```
     git clone https://github.com/Myfriendo/BankTermPredict.git
     ```
   - Navigate to the cloned repository:
     ```
     cd BankTermPredict
     ```

## Running the App
1. **Navigate to the Script Location:**
   Open a terminal or command prompt and navigate to the directory containing `StreamlitApp.py`.

2. **Start the Streamlit App:**
   Run the following command to launch the app:
   ```
   streamlit run StreamlitApp.py
   ```

3. **Access the App:**
   - After running the command, a local URL (e.g., `http://localhost:8501`) will appear in the terminal.
   - Open this URL in your web browser to access the app interface.

## Interacting with the App
1. **Input Customer Details:**
   - The app allows you to input customer attributes such as age, job type, marital status, education, and other financial details.
   - Use the dropdown menus, sliders, and number input fields to provide the necessary data.

2. **Features:**
   - **Age:** Enter the customer’s age.
   - **Job:** Select the job type from the provided options.
   - **Marital Status:** Choose from married, single, or divorced.
   - **Education:** Pick the education level of the customer.
   - **Default, Housing Loan, Personal Loan:** Specify whether the customer has these credit features.
   - **Contact:** Indicate the communication type (e.g., cellular or telephone).
   - **Month & Day of the Week:** Provide the month and day for the campaign.
   - **Numeric Inputs:** Supply numerical values for campaign data (e.g., number of contacts, consumer price index).

3. **Prediction Output:**
   - After filling in the inputs, the app will display whether the customer is likely to subscribe to the term deposit.

## Notes and Troubleshooting
- Ensure the `bank_marketing_pipeline.sav` model file is in the correct directory.
- If you encounter issues with missing libraries, re-run the `pip install` command.
- For further issues, consult the GitHub repository’s README or raise an issue in the repository.

## Deployment
To deploy the app to a public platform like Streamlit Cloud:
1. Commit and push the code to this GitHub repository.
2. Link the repository to Streamlit Cloud.
3. Configure the environment to include the `bank_marketing_pipeline.sav` file and required libraries.
4. Follow Streamlit’s deployment instructions to make the app accessible online.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

