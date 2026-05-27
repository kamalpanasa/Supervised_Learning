from src.preprocessing import (

    load_data,

    preprocess_data,

    split_data
)

from decision_tree_model import (
    train_decision_tree
)

from src.evaluate import (
    evaluate_model
)


# LOAD DATA

df = load_data()


# PREPROCESS DATA

X, y = preprocess_data(df)


# SPLIT DATA

X_train, X_test, y_train, y_test = (
    split_data(X, y)
)


# TRAIN MODEL

model = train_decision_tree(
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
    '\\nDecision Tree Metrics\\n'
)


for key, value in metrics.items():

    print(
        f'{key} :\\n{value}\\n'
    )