import pickle

import numpy as np
import pandas as pd
import streamlit as st

with open('robust_scaler.pkl', 'rb') as file:
    final_scaler = pickle.load(file)

with open('log_reg_model.pkl', 'rb') as file:
    final_model =  pickle.load(file)

def predict_new_transaction(new_data, scaler= final_scaler, model= final_model, threshold=0.17656603277204125):
    """
    Predicts fraud for new transactions using the fitted pipeline and a custom threshold.
    Preserves DataFrame column names to prevent scikit-learn warnings.

    Parameters:
    - new_data: array-like or DataFrame of shape (n_samples, 29) containing V1-V28 and Amount.
    - scaler: The fitted RobustScaler instance.
    - model: The fitted LogisticRegression model.
    - threshold: Custom decision boundary.

    Returns:
    - predictions: List of predicted binary classes (0 or 1).
    - probabilities: List of raw fraud probabilities.
    """
    # Consistently use data_df to preserve feature name alignment
    if isinstance(new_data, pd.DataFrame):
        data_df = new_data.copy()
    else:
        data_arr = np.array(new_data)
        if data_arr.ndim == 1:
            data_arr = data_arr.reshape(1, -1)
        # Recreate a DataFrame with matching column names to avoid warnings
        data_df = pd.DataFrame(data_arr, columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'])

    # 1. Scale features using data_df
    scaled_data = scaler.transform(data_df)


    # 2. Extract probability of fraud (class 1)
    probabilities = model.predict_proba(scaled_data)[:, 1]

    # 3. Apply custom threshold decision rule
    predictions = (probabilities >= threshold).astype(int)

    return predictions, probabilities


st.title("Credit Card Fraud Detection")


V1 = st.number_input("V1", min_value=None, max_value=None, value = -0.674466, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V2 = st.number_input("V2", min_value=None, max_value=None, value = 1.408105, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V3 = st.number_input("V3", min_value=None, max_value=None, value=-1.110622, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V4 = st.number_input("V4", min_value=None, max_value=None, value=-1.328366, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V5 = st.number_input("V5", min_value=None, max_value=None, value=1.388996, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V6 = st.number_input("V6", min_value=None, max_value=None, value=-1.308439, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V7 = st.number_input("V7", min_value=None, max_value=None, value=1.885879, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V8 = st.number_input("V8", min_value=None, max_value=None, value=-0.614233, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V9 = st.number_input("V9", min_value=None, max_value=None, value=0.311652, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V10 = st.number_input("V10", min_value=None, max_value=None, value=0.650757, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V11 = st.number_input("V11", min_value=None, max_value=None, value=-0.857785, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V12 = st.number_input("V12", min_value=None, max_value=None, value=-0.229961, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V13 = st.number_input("V13", min_value=None, max_value=None, value=-0.199817, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V14 = st.number_input("V14", min_value=None, max_value=None, value=0.266371, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V15 = st.number_input("V15", min_value=None, max_value=None, value=-0.046544, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V16 = st.number_input("V16", min_value=None, max_value=None, value=-0.741398, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V17 = st.number_input("V17", min_value=None, max_value=None, value=-0.605617, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V18 = st.number_input("V18", min_value=None, max_value=None, value=-0.392568, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V19 = st.number_input("V19", min_value=None, max_value=None, value=-0.162648, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V20 = st.number_input("V20", min_value=None, max_value=None, value=0.394322, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V21 = st.number_input("V21", min_value=None, max_value=None, value=0.080084, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V22 = st.number_input("V22", min_value=None, max_value=None, value=0.810034, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V23 = st.number_input("V23", min_value=None, max_value=None, value=-0.224327, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V24 = st.number_input("V24", min_value=None, max_value=None, value=0.707899, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V25 = st.number_input("V25", min_value=None, max_value=None, value=-0.135837, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V26 = st.number_input("V26", min_value=None, max_value=None, value=0.045102, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V27 = st.number_input("V27", min_value=None, max_value=None, value=0.533837, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

V28 = st.number_input("V28", min_value=None, max_value=None, value=0.291319, step=None, format=None,
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

Amount = st.number_input("Amount", min_value=None, max_value=None, value=23.000000, step=None, format="%0.4f",
                key=None, help=None, on_change="rerun", args=None, kwargs=None,
                placeholder=None, disabled=False, required=False, label_visibility="visible", 
                icon=None, width="stretch", bind=None, persist_state=None)

arr = np.array([V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
                V21, V22, V23, V24, V25, V26, V27, V28, Amount])

if st.button("Predict"):
    pred, prob = predict_new_transaction(arr)
    st.markdown(f"New Transaction Probability: {prob[0]:.6f}")
    st.markdown(f"Predicted Class (Threshold = {0.176566:.6f}): {pred[0]} ({'Fraud' if pred[0] == 1 else 'Legitimate'})")

