from pathlib import Path

import numpy as np

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent


model_path = (
    BASE_DIR /
    'models' /
    'knn_regressor.pkl'
)


model = joblib.load(
    model_path
)


def predict_price(

    scaler,

    company,

    typename,

    inches,

    ram,

    memory,

    opsys,

    weight
):

    data = np.array([[

        company,

        typename,

        inches,

        ram,

        memory,

        opsys,

        weight

    ]])


    data_scaled = scaler.transform(
        data
    )


    prediction = model.predict(
        data_scaled
    )


    return prediction[0]