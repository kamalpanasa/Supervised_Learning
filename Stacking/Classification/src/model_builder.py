from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier
)


def build_stacking_model():

    base_models = [

        (
            "lr",
            LogisticRegression(
                max_iter=1000
            )
        ),

        (
            "dt",
            DecisionTreeClassifier(
                max_depth=8,
                random_state=42
            )
        ),

        (
            "rf",
            RandomForestClassifier(
                n_estimators=20,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]

    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        n_jobs=-1
    )

    return stacking_model