import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocessing import load_data, preprocess_data
from src.predict import predict_genre


# PAGE CONFIG

st.set_page_config(
    page_title='Spotify Genre Prediction',
    page_icon='🎵',
    layout='wide'
)


# LOAD DATA

df = load_data()
X, y, scaler, genre_encoder = preprocess_data(df)


# CUSTOM CSS

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #A0A0A0;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #1E1E1E;
        text-align: center;
        border: 1px solid #333333;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# HEADER

st.markdown(
    '<div class="main-title">🎵 Spotify Genre Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Predict music genres using K-Nearest Neighbors and Spotify audio features</div>',
    unsafe_allow_html=True
)


# SIDEBAR

st.sidebar.header('About Project')

st.sidebar.info(
    '''
    This application predicts Spotify music genres
    using a KNN Classification model with
    hyperparameter tuning.
    '''
)

st.sidebar.markdown('---')

st.sidebar.write('### Technologies')
st.sidebar.write('- KNN Classifier')
st.sidebar.write('- GridSearchCV')
st.sidebar.write('- Streamlit')
st.sidebar.write('- Scikit-learn')


# MAIN INPUTS

st.subheader('Song Features')

col1, col2 = st.columns(2)


with col1:

    danceability = st.slider(
        'Danceability',
        0.0,
        1.0,
        0.5,
        help='How suitable a track is for dancing'
    )

    energy = st.slider(
        'Energy',
        0.0,
        1.0,
        0.5,
        help='Intensity and activity level'
    )

    acousticness = st.slider(
        'Acousticness',
        0.0,
        1.0,
        0.3,
        help='Confidence measure of acoustic nature'
    )

    valence = st.slider(
        'Valence',
        0.0,
        1.0,
        0.5,
        help='Musical positivity level'
    )


with col2:

    popularity = st.slider(
        'Popularity',
        0,
        100,
        50
    )

    tempo = st.slider(
        'Tempo',
        50,
        220,
        120
    )

    speechiness = st.slider(
        'Speechiness',
        0.0,
        1.0,
        0.2
    )

    instrumentalness = st.slider(
        'Instrumentalness',
        0.0,
        1.0,
        0.1
    )


st.markdown('---')


# PREDICT BUTTON

if st.button('Predict Genre', use_container_width=True):

    prediction = predict_genre(
        scaler,
        genre_encoder,
        popularity,
        200000,
        0,
        danceability,
        energy,
        5,
        -10.0,
        1,
        speechiness,
        acousticness,
        instrumentalness,
        0.2,
        valence,
        tempo,
        4
    )


    st.markdown(
        f'''
        <div class="prediction-box">
            <h2>🎧 Predicted Genre</h2>
            <h1>{prediction.upper()}</h1>
        </div>
        ''',
        unsafe_allow_html=True
    )


    # METRICS

    st.subheader('Audio Feature Summary')

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric('Danceability', danceability)
    metric2.metric('Energy', energy)
    metric3.metric('Valence', valence)
    metric4.metric('Tempo', tempo)


    # FEATURE VISUALIZATION

    st.subheader('Feature Visualization')

    graph_df = pd.DataFrame({
        'Feature': [
            'Danceability',
            'Energy',
            'Acousticness',
            'Speechiness',
            'Instrumentalness',
            'Valence'
        ],
        'Value': [
            danceability,
            energy,
            acousticness,
            speechiness,
            instrumentalness,
            valence
        ]
    })


    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        graph_df['Feature'],
        graph_df['Value']
    )

    ax.set_ylim(0, 1)

    ax.set_ylabel('Feature Value')

    ax.set_title('Spotify Audio Features')

    plt.xticks(rotation=15)

    st.pyplot(fig)


    # RADAR STYLE INFO

    st.subheader('Song Analysis')

    if energy > 0.7:
        st.info('This song has high energy characteristics.')

    if acousticness > 0.7:
        st.info('This song appears highly acoustic.')

    if danceability > 0.7:
        st.info('This song is highly danceable.')

    if instrumentalness > 0.7:
        st.info('This song has strong instrumental characteristics.')


st.markdown('---')

st.caption(
    'Built using Streamlit, KNN Classification, and Spotify audio features.'
)