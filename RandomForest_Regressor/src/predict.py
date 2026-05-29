from pathlib import Path

import numpy as np

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent


model_path = (
    BASE_DIR /
    'models' /
    'random_forest_regressor.pkl'
)


model = joblib.load(
    model_path
)


def predict_house_price(

    bedrooms,

    bathrooms,

    sqft_living,

    floors,

    waterfront,

    view,

    condition,

    grade,

    sqft_above,

    sqft_basement
):

    data = np.array([[

        bedrooms,

        bathrooms,

        sqft_living,

        floors,

        waterfront,

        view,

        condition,

        grade,

        sqft_above,

        sqft_basement

    ]])


    prediction = model.predict(
        data
    )


    return prediction[0]