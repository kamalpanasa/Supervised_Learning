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
        'Attrition.csv'
    )

    df = pd.read_csv(
        data_path,
        encoding='latin1'
    )

    return df


def preprocess_data(df):

    # DROP UNUSED COLUMNS

    df.drop(
        columns=[
            'EmployeeCount',
            'EmployeeNumber',
            'Over18',
            'StandardHours'
        ],
        inplace=True
    )


    # ENCODE TARGET

    df['Attrition'] = (
        df['Attrition']
        .map({
            'Yes': 1,
            'No': 0
        })
    )


    # ENCODE CATEGORICAL FEATURES

    encoder = LabelEncoder()

    categorical_columns = [

        'BusinessTravel',

        'Department',

        'EducationField',

        'Gender',

        'JobRole',

        'MaritalStatus',

        'OverTime'
    ]


    for col in categorical_columns:

        df[col] = encoder.fit_transform(
            df[col]
        )


    # FEATURES

    X = df.drop(
        columns=['Attrition']
    )


    # TARGET

    y = df['Attrition']

    return X, y


def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )