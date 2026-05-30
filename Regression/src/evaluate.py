import joblib
import matplotlib.pyplot as plt

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


def evaluate_model():

    df = load_data(
        "data/raw/train.csv"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = preprocess_data(df)

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)

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

    # Actual vs Predicted
    plt.figure(figsize=(8, 5))

    plt.scatter(
        y_test,
        predictions
    )

    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")

    plt.title("Actual vs Predicted")

    plt.show()


if __name__ == "__main__":
    evaluate_model()