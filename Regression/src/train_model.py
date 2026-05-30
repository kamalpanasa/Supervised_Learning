import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.preprocessing import (
    load_data,
    preprocess_data
)


MODEL_PATH = "models/house_price_model.pkl"


def train_model():

    file_path = "data/raw/train.csv"

    df = load_data(file_path)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = preprocess_data(df)

    model_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
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
    )

    # Train model
    model_pipeline.fit(X_train, y_train)

    # Predictions
    predictions = model_pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model_pipeline,
        MODEL_PATH
    )

    print("Model saved successfully.")