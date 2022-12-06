import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer

title_center = "<h1 style='text-align: center'>WELCOME TO THE APP</h1>"



class FinalPipeline():
    def __init__(self, final_model, scale_cols=None, encode_cols=None):
        self.label_encoders = []
        self.standard_scalers = []
        self.smote = SMOTE()
        self.model = final_model
        self.scale_cols = scale_cols
        self.encode_cols = encode_cols

    def fit(self, X_train, Y_train):
        if self.scale_cols is not None:
            for col in self.scale_cols:
                standard_scaler = StandardScaler()
                X_train[col] = standard_scaler.fit_transform(X_train[col].values.reshape(-1,1))
                self.standard_scalers.append(standard_scaler)
        if self.encode_cols is not None:
            for col in self.encode_cols:
                label_encoder = LabelEncoder()
                X_train[col] = label_encoder.fit_transform(X_train[col])
                self.label_encoders.append(label_encoder)
        X_train_new, Y_train_new = self.smote.fit_resample(X_train, Y_train)
        self.model.fit(X_train_new, Y_train_new)
        return self

    def predict(self, X_test):
        if self.scale_cols is not None:
            for i, col in enumerate(self.scale_cols):
                X_test[col] = self.standard_scalers[i].transform(X_test[col].values.reshape(-1,1))
        if self.encode_cols is not None:
            for i, col in enumerate(self.encode_cols):
                X_test[col] = self.label_encoders[i].transform(X_test[col])
        Y_pred = self.model.predict(X_test)
        return Y_pred


def get_user_input():
    id = st.text_input("Enter your ID")
    gender = st.radio("Your gender", ("Male", "Female"))
    age = st.slider("What's your age?", 0, 100, 18)
    hypertension_string = st.radio("Do you have hypertension?", ("Yes", "No"))
    hypertension = 1 if hypertension_string == "Yes" else 0
    heart_disease_string = st.radio("Do you have heart disease?", ("Yes", "No"))
    heart_disease = 1 if heart_disease_string == "Yes" else 0
    ever_married = st.radio("Are you married?", ("Yes", "No"))
    work_type = st.radio("What's your work type?", ("Private", "Self-employed", "Goverment job", "Children", "Never worked"))
    residence_type = st.radio("What's your residence type?", ("Urban", "Rural"))
    avg_glucose_level = st.slider("What's your average glucose level?", 0.0, 300.0, 100.0)
    weight = st.slider("What's your weight? (kg)", 0.0, 200.0, 50.0, 0.1)
    height = st.slider("What's your height? (m)", 0.0, 2.0, 0.5, 0.01)
    bmi = weight / (height ** 2)
    smoking_status = st.radio("What's your smoking status?", ("Formerly smoked", "Never smoked", "Smokes", "Unknown"))
    data = {
        "id": id,
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }
    record = pd.DataFrame(data, index=[0])
    return record


def get_prediction(record, pipeline):
    input = record.drop(columns=['id', 'ever_married', 'work_type', 'residence_type', 'smoking_status'])
    prediction = pipeline.predict(input)
    return prediction


def main():
    pipeline = joblib.load('pipeline.gz')
    st.title("Stroke Prediction App")
    record = get_user_input()
    if st.button("Predict"):
        with st.spinner('Wait for it...'):
            time.sleep(2.5)
        prediction = get_prediction(record, pipeline)
        if prediction == 1:
            st.error("You have a HIGH risk of stroke!")
        if prediction == 0:
            st.success("You have a low risk of stroke")


if __name__ == "__main__":
    main()

# Press Ctrl + C in the terminal to stop the web