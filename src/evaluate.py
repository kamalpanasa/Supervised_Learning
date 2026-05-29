from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    classification_report,

    roc_auc_score
)


def evaluate_model(

    model,

    X_test,

    y_test
):

    predictions = model.predict(
        X_test
    )


    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )


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


    roc_auc = roc_auc_score(
        y_test,
        probabilities
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

        'ROC-AUC': roc_auc,

        'Confusion Matrix': matrix,

        'Classification Report': report
    }