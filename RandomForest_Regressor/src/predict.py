from pathlib import Path

import numpy as np

import joblib

from src.preprocessing import (
    load_data,
    preprocess_data,
    split_data
)

from src.random_forest_regressor import (
    train_random_forest_regressor
)


BASE_DIR = Path(__file__).resolve().parent.parent


model_path = (
    BASE_DIR /
    'models' /
    'random_forest_regressor.pkl'
)


# AUTO TRAIN MODEL IF NOT FOUND

if not model_path.exists():

    print(
        'Model not found. Training model...'
    )

    df = load_data()

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = (
        split_data(X, y)
    )

    model = train_random_forest_regressor(
        X_train,
        y_train
    )

else:

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