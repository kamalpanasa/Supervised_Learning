from pathlib import Path

from sklearn.ensemble import (
    RandomForestRegressor
)

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent


def train_random_forest_regressor(
    X_train,
    y_train
):

    model = RandomForestRegressor(

        n_estimators=20,

        max_depth=10,

        min_samples_split=5,

        min_samples_leaf=2,

        random_state=42,

        n_jobs=-1
    )


    print(
        'Training Started...',
        flush=True
    )


    model.fit(
        X_train,
        y_train
    )


    print(
        'Training Completed...',
        flush=True
    )


    model_path = (
        BASE_DIR /
        'models' /
        'random_forest_regressor.pkl'
    )


    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    joblib.dump(
        model,
        model_path
    )


    return model