import time
import joblib
import pandas as pd
import streamlit as st
import scipy.stats as stats
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn import metrics


def get_user_input():
    id = st.text_input("Enter your ID")
    gender = st.radio("Your gender", ("Male", "Female"))
    age = st.slider("What's your age?", 0, 100, 18)
    hypertension_string = st.radio("Do you have hypertension?", ("Yes", "No"))
    hypertension = 1 if hypertension_string == "Yes" else 0
    heart_disease_string = st.radio("Do you have heart disease?", ("Yes", "No"))
    heart_disease = 1 if heart_disease_string == "Yes" else 0
    ever_married = st.radio("Are you married?", ("Yes", "No"))
    work_type = st.radio("What's your work type?", ("Private", "Self-employed", "Goverment job", "Never worked"))
    residence_type = st.radio("What's your residence type?", ("Urban", "Rural"))
    avg_glucose_level = st.slider("What's your average glucose level?", 0.0, 300.0, 100.0)
    weight = st.slider("What's your weight? (kg)", 0.0, 200.0, 50.0, 0.1)
    height = st.slider("What's your height? (m)", 0.0, 2.0, 0.5, 0.01)
    bmi = weight / (height ** 2)
    smoking_status = st.radio("What's your smoking status?", ("Formerly Smoked", "Never Smoked", "Smokes"))
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

def convert_string(record):
    record = record.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    record = record.replace(' ', '_', regex=True).replace('-', '_', regex=True)
    print(record.dtypes)
    print(record)
    return record

def get_prediction(record, pipeline):
    input = record.drop(columns=['id'])
    prediction = pipeline.predict(input)
    return prediction

def f2(y_true, y_pred):
    return metrics.fbeta_score(y_true, y_pred, beta=2)

f2_score = metrics.make_scorer(f2, greater_is_better=True)

def main():
    pipeline = joblib.load("scripts/pipeline.pkl")
    st.title("Stroke Prediction App")
    record = get_user_input()
    record = convert_string(record)
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