from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    LabelEncoder
)


BASE_DIR = Path(__file__).resolve().parent.parent


def load_data():

    data_path = (
        BASE_DIR /
        'data' /
        'raw' /
        'insurance.csv'
    )

    df = pd.read_csv(data_path)

    return df


def preprocess_data(df):

    # ENCODE CATEGORICAL FEATURES

    encoder = LabelEncoder()

    categorical_columns = [

        'sex',

        'smoker',

        'region'
    ]


    for col in categorical_columns:

        df[col] = encoder.fit_transform(
            df[col]
        )


    # FEATURES

    X = df.drop(
        columns=['charges']
    )


    # TARGET

    y = df['charges']

    return X, y


def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )