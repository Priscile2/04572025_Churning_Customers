import streamlit as st
import joblib
import pandas as pd
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model('best_model.pkl')

st.title("Customer Churn Prediction App: Priscile Nkenmeza")

# Input fields for user data
tenure = st.slider("Tenure (months)", min_value=0, max_value=72)
monthly_charges = st.number_input("Monthly Charges ($)")
total_charges = st.number_input("Total Charges ($)")
online_security = st.radio("Online Security", ["No", "Yes", "No Internet Service"])
device_protection = st.radio("Device Protection", ["No", "Yes", "No Internet Service"])
tech_support = st.radio("Tech Support", ["No", "Yes", "No Internet Service"])
streaming_tv = st.radio("Streaming TV", ["No", "Yes", "No Internet Service"])
streaming_movies = st.radio("Streaming Movies", ["No", "Yes", "No Internet Service"])
contract = st.radio("Contract", ["Month-to-Month", "One Year", "Two Year"])
payment_method = st.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer (automatic)", "Credit Card (automatic)"])

# Button for making predictions
if st.button("Predict Churn"):
    # Prepare the input data for prediction
    input_data = [[tenure, monthly_charges, total_charges, online_security, device_protection, tech_support, streaming_tv, streaming_movies, contract, payment_method]]

    input_data = pd.DataFrame(input_data)
    
    num_data= input_data.select_dtypes(include=['int64','float64'])
    cat_data=input_data.select_dtypes(exclude=['int64','float64'])
    
    names = list(cat_data.columns.values)

    for column in names:
        cat_data[column], _ = pd.factorize(cat_data[column])

    from sklearn.preprocessing import StandardScaler

    scale = StandardScaler()
    x_scale = scale.fit_transform(num_data)


    # Assuming 'X_scaled' is your scaled NumPy array and 'X' is your original DataFrame
    num_data = pd.DataFrame(x_scale, columns=num_data.columns)
    
    data2 = pd.concat([num_data, cat_data], axis=1)
    
    # Make a prediction using the loaded model
    prediction = loaded_model.predict(data2)

    # Display the prediction
    churn_prediction = (1-(prediction[0][0]))*100
    st.write(f"There is a  :{churn_prediction} % confidence that... ")
    if prediction[0][0] == 1:
        st.write("This customer is likely to churn.")
    else:
        st.write("This customer is not likely to churn.")
