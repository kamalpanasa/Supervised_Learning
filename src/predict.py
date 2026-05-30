from pathlib import Path

import numpy as np

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent


model_path = (
    BASE_DIR /
    'models' /
    'random_forest_classifier.pkl'
)


model = joblib.load(
    model_path
)


def predict_attack(
    features
):

    data = np.array([
        features
    ])


    prediction = model.predict(
        data
    )


    probability = (
        model.predict_proba(data)
    )


    return prediction[0], probability[0]