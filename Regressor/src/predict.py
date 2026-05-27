from pathlib import Path

import numpy as np

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent


model_path = (
    BASE_DIR /
    'models' /
    'decision_tree_regressor.pkl'
)


model = joblib.load(
    model_path
)


def predict_insurance_cost(

    age,

    sex,

    bmi,

    children,

    smoker,

    region
):

    data = np.array([[

        age,

        sex,

        bmi,

        children,

        smoker,

        region

    ]])


    prediction = model.predict(
        data
    )


    return prediction[0]