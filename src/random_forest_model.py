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
            50,
            100
        ],

        'max_depth': [
            10,
            20
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

        scoring='f1',

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


    model_path = (
        BASE_DIR /
        'models' /
        'random_forest_model.pkl'
    )


    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    joblib.dump(
        best_model,
        model_path
    )


    return best_model