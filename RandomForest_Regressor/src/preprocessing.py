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
        'kc_house_data.csv'
    )

    df = pd.read_csv(
        data_path,
        encoding='latin1'
    )

    return df


def preprocess_data(df):

    # DROP UNUSED COLUMNS

    drop_columns = [

        'date',

        'id'
    ]


    for col in drop_columns:

        if col in df.columns:

            df.drop(
                columns=[col],
                inplace=True
            )


    # FEATURES

    X = df.drop(
        columns=['price']
    )


    # TARGET

    y = df['price']


    # ENCODE CATEGORICAL FEATURES

    encoder = LabelEncoder()


    for col in X.columns:

        if X[col].dtype == 'object':

            X[col] = encoder.fit_transform(
                X[col]
            )


    return X, y


def split_data(X, y):

    return train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42
    )