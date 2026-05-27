from pathlib import Path

from sklearn.neighbors import (
    KNeighborsRegressor
)

from sklearn.model_selection import (
    GridSearchCV
)

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent


def train_knn_regressor(
    X_train,
    y_train
):

    model = KNeighborsRegressor()


    param_grid = {

        'n_neighbors': [
            3,
            5,
            7,
            9
        ],

        'weights': [
            'uniform',
            'distance'
        ],

        'metric': [
            'euclidean',
            'manhattan',
            'minkowski'
        ]
    }


    grid_search = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        cv=5,

        scoring='r2'
    )


    grid_search.fit(
        X_train,
        y_train
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
        'knn_regressor.pkl'
    )


    joblib.dump(
        best_model,
        model_path
    )


    return best_model