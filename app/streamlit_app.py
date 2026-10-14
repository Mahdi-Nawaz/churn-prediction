import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '../models/churn_model.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

st.title("🔮 Telecom Customer Churn Prediction")
st.markdown("Enter the customer's details below to predict their probability of churning.")

if model is None:
    st.error("Model not found! Please ensure you have run the `03_modeling.ipynb` notebook to generate and save `models/churn_model.joblib`.")
    st.stop()

# Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    st.subheader("Account Information")
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)

with col3:
    st.subheader("Services Subscribed")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    # Conditional display based on internet service
    if internet_service != "No":
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    else:
        online_security = "No"
        online_backup = "No"
        device_protection = "No"
        tech_support = "No"
        streaming_tv = "No"
        streaming_movies = "No"

# Prediction button
if st.button("Predict Churn Risk", type="primary"):
    
    # Create input dataframe
    input_data = {
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    }
    
    df_input = pd.DataFrame(input_data)
    
    # Predict
    try:
        churn_prob = model.predict_proba(df_input)[0][1]
        prediction = "Churn" if churn_prob > 0.5 else "Stay"
        
        st.divider()
        st.subheader("Prediction Results")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if prediction == "Churn":
                st.error(f"**High Risk of Churn**")
            else:
                st.success(f"**Likely to Stay**")
                
            st.metric(label="Churn Probability", value=f"{churn_prob:.1%}")
            
        with col_res2:
            st.markdown("### Top Factors for this Customer")
            st.write("*(Note: To generate actual SHAP explanations for real-time inference, the preprocessor step in the pipeline must be separated to fit the SHAP explainer).*")
            
            # Simple heuristic explanation since SHAP in real-time requires the explainer
            st.write(f"- Contract type: {contract}")
            st.write(f"- Tenure: {tenure} months")
            st.write(f"- Internet Service: {internet_service}")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Check if the categorical values exactly match what the model expects.")
