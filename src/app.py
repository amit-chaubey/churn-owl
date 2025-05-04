"""
Streamlit web application for Customer Churn Prediction.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.utils.visualization import plot_feature_importance

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = Path("models/customer_churn_model.joblib")
    if not model_path.exists():
        st.error("Model file not found. Please train the model first.")
        return None
    return joblib.load(model_path)

def main():
    # Add header
    st.title("📊 Customer Churn Predictor")
    st.write("""
    Predict whether a customer is likely to churn based on their characteristics.
    This model uses historical customer data to make predictions.
    """)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Input form
    with col1:
        st.subheader("Customer Information")
        
        # Categorical Features
        st.write("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        st.write("**Services**")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        st.write("**Contract Details**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        st.write("**Usage & Charges**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 0, 200, 50)
        total_charges = st.number_input("Total Charges ($)", 0, 10000, value=tenure * monthly_charges)

    # Create a dictionary of features
    features = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Create DataFrame from features
    input_df = pd.DataFrame([features])
    
    # Make prediction when button is clicked
    with col2:
        st.subheader("Prediction")
        if st.button("Predict Churn Probability"):
            model_dict = load_model()
            if model_dict is not None:
                # Get model and preprocessor
                model = model_dict['model']
                preprocessor = model_dict['preprocessor']
                
                # Preprocess the input
                X = preprocessor.transform(input_df)
                
                # Make prediction
                churn_prob = model.predict_proba(X)[0, 1]
                
                # Display prediction
                st.write("### Churn Probability")
                
                # Create a progress bar for the probability
                st.progress(churn_prob)
                
                # Display the probability as a percentage
                st.write(f"### {churn_prob:.1%}")
                
                # Add interpretation
                if churn_prob < 0.3:
                    st.success("🟢 Low risk of churn")
                elif churn_prob < 0.6:
                    st.warning("🟡 Moderate risk of churn")
                else:
                    st.error("🔴 High risk of churn")
                
                # Feature importance
                st.write("### Top Factors Influencing Prediction")
                feature_names = (
                    preprocessor.named_transformers_['cat']
                    .named_steps['onehot']
                    .get_feature_names_out(CATEGORICAL_FEATURES)
                )
                feature_names = list(feature_names) + NUMERICAL_FEATURES
                
                importances = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                })
                importances = importances.sort_values('importance', ascending=False).head(5)
                
                st.bar_chart(importances.set_index('feature'))
                
                # Add recommendations based on top factors
                st.write("### Recommendations")
                st.write("""
                Based on the prediction, here are some recommendations:
                - Monitor customer satisfaction regularly
                - Offer personalized retention deals
                - Provide proactive customer support
                - Consider contract upgrades or special promotions
                """)
            else:
                st.error("Please train the model first using the main script.")

if __name__ == "__main__":
    main() 