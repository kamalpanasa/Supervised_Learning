from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.model_selection import (
    GridSearchCV
)

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent


def train_random_forest(
    X_train,
    y_train
):

    model = RandomForestClassifier(

        random_state=42,

        class_weight='balanced'
    )


    param_grid = {

        'n_estimators': [
            100,
            200
        ],

        'max_depth': [
            10,
            20,
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

        scoring='f1'
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
        '\\nBest F1 Score:\\n'
    )

    print(
        grid_search.best_score_
    )


    model_path = (
        BASE_DIR /
        'models' /
        'random_forest_model.pkl'
    )


    joblib.dump(
        best_model,
        model_path
    )


    return best_model