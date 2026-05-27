from src.preprocessing import (

    load_data,

    preprocess_data,

    split_data
)

from src.knn_regressor import (
    train_knn_regressor
)

from src.evaluate import (
    evaluate_model
)


# LOAD DATA

df = load_data()


# PREPROCESS DATA

X, y, scaler = preprocess_data(df)


# SPLIT DATA

X_train, X_test, y_train, y_test = (
    split_data(X, y)
)


# TRAIN MODEL

model = train_knn_regressor(
    X_train,
    y_train
)


# EVALUATE MODEL

metrics = evaluate_model(

    model,

    X_test,

    y_test
)


print(
    '\\nKNN Regression Metrics\\n'
)


for key, value in metrics.items():

    print(
        f'{key} : {value}\\n'
    )