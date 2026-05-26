from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

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
        'spotify.csv'
    )

    df = pd.read_csv(data_path)

    return df


def preprocess_data(df):

    # KEEP ONLY TOP GENRES

    top_genres = [
        'pop',
        'rock',
        'hip-hop',
        'classical',
        'jazz'
    ]

    df = df[
        df['track_genre'].isin(top_genres)
    ]


    # DROP UNUSED COLUMNS

    df.drop(
        columns=[
            'S.NO',
            'track_id',
            'artists',
            'album_name',
            'track_name'
        ],
        inplace=True
    )


    # ENCODE BOOLEAN COLUMN

    df['explicit'] = (
        df['explicit']
        .astype(int)
    )


    # ENCODE TARGET

    genre_encoder = LabelEncoder()

    df['track_genre'] = (
        genre_encoder.fit_transform(
            df['track_genre']
        )
    )


    # FEATURES

    X = df.drop(
        columns=['track_genre']
    )


    # TARGET

    y = df['track_genre']


    # SCALE FEATURES

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return (
        X_scaled,
        y,
        scaler,
        genre_encoder
    )


def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )