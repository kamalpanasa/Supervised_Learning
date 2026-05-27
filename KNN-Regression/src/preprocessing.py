from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)


BASE_DIR = Path(__file__).resolve().parent.parent


def load_data():

    data_path = (
        BASE_DIR /
        'data' /
        'raw' /
        'laptop.csv'
    )

    df = pd.read_csv(
    data_path,
    encoding='latin1'
)

    return df


def preprocess_data(df):

    # DROP UNUSED COLUMN

    df.drop(
        columns=['laptop_ID'],
        inplace=True
    )


    # SELECT IMPORTANT FEATURES

    selected_columns = [

        'Company',

        'TypeName',

        'Inches',

        'Ram',

        'Memory',

        'OpSys',

        'Weight',

        'Price_euros'
    ]

    df = df[selected_columns]


    # CLEAN RAM COLUMN

    df['Ram'] = (
        df['Ram']
        .str.replace('GB', '')
        .astype(int)
    )


    # CLEAN WEIGHT COLUMN

    df['Weight'] = (
        df['Weight']
        .str.replace('kg', '')
        .astype(float)
    )


    # ENCODE CATEGORICAL FEATURES

    encoder = LabelEncoder()

    categorical_columns = [

        'Company',

        'TypeName',

        'Memory',

        'OpSys'
    ]


    for col in categorical_columns:

        df[col] = encoder.fit_transform(
            df[col]
        )


    # FEATURES

    X = df.drop(
        columns=['Price_euros']
    )


    # TARGET

    y = df['Price_euros']


    # FEATURE SCALING

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )