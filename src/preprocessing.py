from pathlib import Path

import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder
)


BASE_DIR = Path(__file__).resolve().parent.parent


def load_data():

    train_path = (
        BASE_DIR /
        'data' /
        'raw' /
        'Train_data.csv'
    )

    test_path = (
        BASE_DIR /
        'data' /
        'raw' /
        'Test_data.csv'
    )


    train_df = pd.read_csv(
        train_path
    )

    test_df = pd.read_csv(
        test_path
    )

    return train_df, test_df


def preprocess_data(
    train_df,
    test_df
):

    combined_df = pd.concat(
        [train_df, test_df],
        axis=0
    )


    categorical_columns = [

        'protocol_type',

        'service',

        'flag'
    ]


    encoder = LabelEncoder()


    for col in categorical_columns:

        combined_df[col] = (
            encoder.fit_transform(
                combined_df[col]
            )
        )


        combined_df['class'] = (
            combined_df['class']
                .apply(
                lambda x: 0 if x == 'normal' else 1
            )
        )


    train_processed = (
        combined_df.iloc[
            :len(train_df)
        ]
    )


    test_processed = (
        combined_df.iloc[
            len(train_df):
        ]
    )


    X_train = train_processed.drop(
        columns=['class']
    )

    y_train = train_processed['class']


    X_test = test_processed.drop(
        columns=['class']
    )

    y_test = test_processed['class']


    return (

        X_train,

        X_test,

        y_train,

        y_test
    )