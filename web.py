import time
import joblib
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder

title_center = "<h1 style='text-align: center'>WELCOME TO THE APP</h1>"



class FinalPipeline():
    def __init__(self, final_model):
        self.label_encoder = LabelEncoder()
        self.standard_scaler = StandardScaler()
        self.smote = SMOTE()
        self.model = final_model

    def fit(self, X_train, Y_train):
        scale_cols = ['bmi', 'avg_glucose_level', 'age']
        for col in scale_cols:
            X_train[col] = self.standard_scaler.fit_transform(X_train[col].values.reshape(-1,1))
        X_train['gender'] = self.label_encoder.fit_transform(X_train['gender'])
        X_train_new, Y_train_new = self.smote.fit_resample(X_train, Y_train)
        self.model.fit(X_train_new, Y_train_new)
        return self

    def predict(self, X_test):
        scale_cols = ['bmi', 'avg_glucose_level', 'age']
        for col in scale_cols:
            X_test[col] = self.standard_scaler.transform(X_test[col].values.reshape(-1,1))
        X_test['gender'] = self.label_encoder.transform(X_test['gender'])
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
    pipeline = joblib.load('scripts/pipeline.gz')
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