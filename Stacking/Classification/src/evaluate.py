import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)

from src.preprocessing import (
    load_data,
    preprocess_data
)


MODEL_PATH = "models/stacking_loan_model.pkl"


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

    probabilities = model.predict_proba(X_test)[:, 1]

    # Classification Report
    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    # ROC AUC
    roc_score = roc_auc_score(
        y_test,
        probabilities
    )

    print(f"ROC-AUC Score: {roc_score:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

    # ROC Curve
    RocCurveDisplay.from_predictions(
        y_test,
        probabilities
    )

    plt.title("ROC Curve")

    plt.show()


if __name__ == "__main__":
    evaluate_model()