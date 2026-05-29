from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split
)


BASE_DIR = Path(__file__).resolve().parent.parent


def load_data():

    data_path = (
        BASE_DIR /
        'data' /
        'raw' /
        'creditcard.csv'
    )

    df = pd.read_csv(
        data_path
    )

    return df


def preprocess_data(df):

    # FEATURES

    X = df.drop(
        columns=['Class']
    )


    # TARGET

    y = df['Class']


    return X, y


def split_data(X, y):

    return train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )