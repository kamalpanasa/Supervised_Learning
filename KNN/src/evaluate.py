from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    classification_report
)


def evaluate_model(

    model,

    X_test,

    y_test
):

    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions,
        average='weighted'
    )


    recall = recall_score(
        y_test,
        predictions,
        average='weighted'
    )


    f1 = f1_score(
        y_test,
        predictions,
        average='weighted'
    )


    matrix = confusion_matrix(
        y_test,
        predictions
    )


    report = classification_report(
        y_test,
        predictions
    )


    return {

        'Accuracy': accuracy,

        'Precision': precision,

        'Recall': recall,

        'F1 Score': f1,

        'Confusion Matrix': matrix,

        'Classification Report': report
    }