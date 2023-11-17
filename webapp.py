import streamlit as st
import pandas as pd
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
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

input_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'OnlineSecurity': online_security,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaymentMethod':payment_method,
}

# Button for making predictions
if st.button("Predict Churn"):
    # Prepare the input data for prediction
    #input_data = [[tenure, monthly_charges, total_charges, online_security, device_protection, tech_support, streaming_tv, streaming_movies, contract, payment_method]]

    input_data = pd.DataFrame(input_data, index=[0])
    

    num_data= input_data.select_dtypes(include=['int64','float64'])
    cat_data=input_data.select_dtypes(exclude=['int64','float64'])
    
    print()
    # Mapping categorical values to numerical
    online_security_mapping = {'No': 0, 'Yes': 1, 'No Internet Service': 2}
    cat_data['OnlineSecurity'] = cat_data['OnlineSecurity'].map(online_security_mapping)

    device_protection_mapping = {'No': 0, 'Yes': 1, 'No Internet Service': 2}
    cat_data['DeviceProtection'] = cat_data['DeviceProtection'].map(device_protection_mapping)

    tech_support_mapping = {'No': 0, 'Yes': 1, 'No Internet Service': 2}
    cat_data['TechSupport'] = cat_data['TechSupport'].map(tech_support_mapping)

    streaming_tv_mapping = {'No': 0, 'Yes': 1, 'No Internet Service': 2}
    cat_data['StreamingTV'] = cat_data['StreamingTV'].map(streaming_tv_mapping) 
    streaming_movies_mapping = {'No': 0, 'Yes': 1, 'No Internet Service': 2}
    cat_data['StreamingMovies'] = cat_data['StreamingMovies'].map(streaming_movies_mapping)

    contract_mapping = {'Month-to-Month': 0, 'One Year': 1, 'Two Year': 2}
    cat_data['Contract'] = cat_data['Contract'].map(contract_mapping)

    payment_method_mapping = {
        'Electronic Check': 0, 'Mailed Check': 1, 'Bank Transfer (automatic)': 2, 'Credit Card (automatic)': 3
    }
    cat_data['PaymentMethod'] = cat_data['PaymentMethod'].map(payment_method_mapping)

    #print(input_data)


   
    


    print("\n categorical------")
    for i in cat_data:
        print(i)
    
    from sklearn.preprocessing import StandardScaler
    
    with open('scaler.pkl', 'rb') as scale:
        scaler = pickle.load(scale)

    x_scale = scaler.transform(num_data)


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
