import streamlit as st
import pandas as pd
import joblib
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the pre-trained model
model_path = 'random_forest_model.pkl'
model = joblib.load(model_path)

# Preprocessing function
def preprocess(data):
    """
    Applies Frequency Encoding to categorical variables and Binary Encoding to binary variables.
    Also transforms date of birth into age.

    Parameters:
    data (pd.DataFrame): The input DataFrame with categorical and date of birth columns.

    Returns:
    pd.DataFrame: The transformed and encoded DataFrame.
    """
    # Frequency Encoding for 'category', 'city', and 'state'
    if 'category' in data.columns:
        data['category'] = data['category'].map(data['category'].value_counts(normalize=True))
    if 'city' in data.columns:
        data['city'] = data['city'].map(data['city'].value_counts(normalize=True))
    if 'state' in data.columns:
        data['state'] = data['state'].map(data['state'].value_counts(normalize=True))

    # Binary Encoding for 'gender'
    if 'gender' in data.columns:
        data['gender'] = data['gender'].map({'F': 0, 'M': 1})

    # Transform 'dob' into 'age' if 'dob' column exists
    if 'dob' in data.columns:
        data['dob'] = pd.to_datetime(data['dob'], errors='coerce')
        current_year = pd.to_datetime('today').year
        data['age'] = current_year - data['dob'].dt.year

    # Drop the original 'dob' column as it has been converted to 'age'
    if 'dob' in data.columns:
        data = data.drop('dob', axis=1)

    # Identify numerical and categorical columns
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Scale numerical columns
    scaler = StandardScaler()
    if len(numerical_columns) > 0:
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Encode categorical columns
    encoder = ce.TargetEncoder(cols=categorical_columns)
    if len(categorical_columns) > 0:
        data = encoder.fit_transform(data, np.zeros(len(data)))  # Dummy target variable

    # Check and handle missing or infinite values
    if data.isna().sum().sum() > 0:
        data = data.fillna(0)
    if np.isinf(data).sum().sum() > 0:
        data = data.replace([np.inf, -np.inf], 0)

    return data

def main():
    st.title('Fraud Detection Prediction App')

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.write(data.head())

        if st.button('Predict'):
            X_encoded = preprocess(data)
        if 'is_fraud' in X_encoded.columns:
            X_encoded = X_encoded.drop(columns=['is_fraud'])

            # Make predictions
            predictions = model.predict(X_encoded)
            prediction_prob = model.predict_proba(X_encoded)[:, 1]

            result_df = data.copy()
            result_df['predicted_fraud'] = predictions
            result_df['fraud_probability'] = prediction_prob

            st.write("### Predictions")
            st.write(result_df.head())

            # Downloadable CSV
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )

# Run the app
if __name__ == '__main__':
    main()
