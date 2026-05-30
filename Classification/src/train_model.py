import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

from src.preprocessing import (
    load_data,
    preprocess_data
)


def train_model():

    file_path = "data/raw/Telco-Customer-Churn.csv"

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
                AdaBoostClassifier(
                    n_estimators=50,
                    learning_rate=0.5,
                    random_state=42
                )
            )
        ]
    )

    # Train model
    model_pipeline.fit(X_train, y_train)

    # Predictions
    predictions = model_pipeline.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"Accuracy: {accuracy:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model_pipeline,
        "models/adaboost_churn_model.pkl"
    )

    print("Model saved successfully.")