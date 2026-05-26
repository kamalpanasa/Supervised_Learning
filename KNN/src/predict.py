from pathlib import Path

import numpy as np

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent


model_path = (
    BASE_DIR /
    'models' /
    'knn_model.pkl'
)


model = joblib.load(
    model_path
)


def predict_genre(

    scaler,

    genre_encoder,

    popularity,

    duration_ms,

    explicit,

    danceability,

    energy,

    key,

    loudness,

    mode,

    speechiness,

    acousticness,

    instrumentalness,

    liveness,

    valence,

    tempo,

    time_signature
):

    data = np.array([[

        popularity,

        duration_ms,

        explicit,

        danceability,

        energy,

        key,

        loudness,

        mode,

        speechiness,

        acousticness,

        instrumentalness,

        liveness,

        valence,

        tempo,

        time_signature

    ]])


    data_scaled = scaler.transform(
        data
    )


    prediction = model.predict(
        data_scaled
    )


    genre = (
        genre_encoder.inverse_transform(
            prediction
        )
    )


    return genre[0]