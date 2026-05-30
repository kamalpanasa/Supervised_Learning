from src.preprocessing import (

    load_data,

    preprocess_data
)

from src.random_forest_model import (
    train_random_forest
)

from src.evaluate import (
    evaluate_model
)


# LOAD DATA

train_df, test_df = load_data()


# PREPROCESS

X_train, X_test, y_train, y_test = (
    preprocess_data(
        train_df,
        test_df
    )
)


# TRAIN MODEL

model = train_random_forest(
    X_train,
    y_train
)


# EVALUATE

metrics = evaluate_model(

    model,

    X_test,

    y_test
)


print(
    '\nRandom Forest Classification Metrics\n'
)


for key, value in metrics.items():

    print(
        f'{key} :\n{value}\n'
    )