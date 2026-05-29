from pathlib import Path

from sklearn.ensemble import (
    RandomForestRegressor
)

from sklearn.model_selection import (
    GridSearchCV
)

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent


def train_random_forest_regressor(
    X_train,
    y_train
):

    model = RandomForestRegressor(
        random_state=42
    )


    param_grid = {

        'n_estimators': [
            50,
            100
        ],

        'max_depth': [
            5,
            10,
            None
        ],

        'min_samples_split': [
            2,
            5
        ],

        'min_samples_leaf': [
            1,
            2
        ]
    }


    grid_search = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        cv=3,

        scoring='r2',

        n_jobs=-1,

        verbose=2
    )


    print(
        'Training Started...',
        flush=True
    )


    grid_search.fit(
        X_train,
        y_train
    )


    print(
        'Training Completed...',
        flush=True
    )


    best_model = (
        grid_search.best_estimator_
    )


    print(
        '\\nBest Parameters:\\n'
    )

    print(
        grid_search.best_params_
    )


    print(
        '\\nBest R2 Score:\\n'
    )

    print(
        grid_search.best_score_
    )


    model_path = (
        BASE_DIR /
        'models' /
        'random_forest_regressor.pkl'
    )


    joblib.dump(
        best_model,
        model_path
    )


    return best_model