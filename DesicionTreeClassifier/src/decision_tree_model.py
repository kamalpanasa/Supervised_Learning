from pathlib import Path

from sklearn.tree import (
    DecisionTreeClassifier
)

from sklearn.model_selection import (
    GridSearchCV
)

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent


def train_decision_tree(
    X_train,
    y_train
):

    model = DecisionTreeClassifier(
        random_state=42
    )


    param_grid = {

        'criterion': [
            'gini',
            'entropy'
        ],

        'max_depth': [
            3,
            5,
            10,
            None
        ],

        'min_samples_split': [
            2,
            5,
            10
        ],

        'min_samples_leaf': [
            1,
            2,
            4
        ]
    }


    grid_search = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        cv=5,

        scoring='accuracy'
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
        '\\nBest Accuracy:\\n'
    )

    print(
        grid_search.best_score_
    )


    model_path = (
        BASE_DIR /
        'models' /
        'decision_tree_model.pkl'
    )


    joblib.dump(
        best_model,
        model_path
    )


    return best_model