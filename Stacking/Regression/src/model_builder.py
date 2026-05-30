from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    StackingRegressor
)


def build_stacking_model():

    base_models = [

        (
            "lr",
            LinearRegression()
        ),

        (
            "dt",
            DecisionTreeRegressor(
                max_depth=8,
                random_state=42
            )
        ),

        (
            "rf",
            RandomForestRegressor(
                n_estimators=20,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=LinearRegression(),
        n_jobs=-1
    )

    return stacking_model