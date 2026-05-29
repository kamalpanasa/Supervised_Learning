from pathlib import Path

import numpy as np

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent

model_path = (
    BASE_DIR /
    'models' /
    'random_forest_regressor.pkl'
)

try:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Run `python main.py` from the RandomForest_Regressor folder to train and save the model."
        )

    model = joblib.load(model_path)
except Exception as exc:
    raise RuntimeError(
        f"Unable to load Random Forest model from {model_path}. "
        "Make sure the model has been trained and the file is not corrupted. "
        "Run `python main.py` from the RandomForest_Regressor folder to create the model file."
    ) from exc


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