import os
import joblib
import pandas as pd

from src.train_model import train_model


MODEL_PATH = "models/adaboost_churn_model.pkl"


def load_model():

    if not os.path.exists(MODEL_PATH):

        print("Model not found.")
        print("Training model automatically...")

        train_model()

    model = joblib.load(MODEL_PATH)

    return model


def predict_churn(input_data):

    model = load_model()

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability