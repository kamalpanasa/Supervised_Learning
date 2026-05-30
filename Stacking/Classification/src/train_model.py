import os
import joblib

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.preprocessing import (
    load_data,
    preprocess_data
)

from src.model_builder import (
    build_stacking_model
)


MODEL_PATH = "models/stacking_loan_model.pkl"


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
                build_stacking_model()
            )
        ]
    )

    # Train model
    model_pipeline.fit(X_train, y_train)

    # Predictions
    predictions = model_pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model_pipeline,
        MODEL_PATH
    )

    print("Model saved successfully.")