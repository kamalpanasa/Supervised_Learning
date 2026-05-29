import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.predict import (
    predict_house_price
)


# PAGE CONFIG

st.set_page_config(
    page_title='House Price Prediction',
    page_icon='🏠',
    layout='wide'
)


# HEADER

st.title(
    '🏠 House Price Prediction'
)

st.write(
    '''
    Predict house prices using
    Random Forest Regression.
    '''
)


# SIDEBAR

st.sidebar.header(
    'Project Information'
)

st.sidebar.info(
    '''
    This application predicts
    house prices based on
    house specifications.
    '''
)


# INPUT SECTION

st.subheader(
    'House Specifications'
)

col1, col2 = st.columns(2)


with col1:

    bedrooms = st.slider(
        'Bedrooms',
        1,
        10,
        3
    )

    bathrooms = st.slider(
        'Bathrooms',
        1.0,
        10.0,
        2.0
    )

    sqft_living = st.slider(
        'Living Area (sqft)',
        500,
        10000,
        2000
    )

    floors = st.slider(
        'Floors',
        1,
        5,
        2
    )

    waterfront = st.selectbox(
        'Waterfront',
        [0, 1]
    )


with col2:

    view = st.slider(
        'View Rating',
        0,
        5,
        2
    )

    condition = st.slider(
        'Condition',
        1,
        5,
        3
    )

    grade = st.slider(
        'Grade',
        1,
        13,
        7
    )

    sqft_above = st.slider(
        'Sqft Above',
        500,
        10000,
        1500
    )

    sqft_basement = st.slider(
        'Sqft Basement',
        0,
        5000,
        500
    )


st.markdown('---')


# PREDICTION

if st.button(
    'Predict House Price',
    use_container_width=True
):

    prediction = (
        predict_house_price(

            bedrooms,

            bathrooms,

            sqft_living,

            floors,

            waterfront,

            view,

            condition,

            grade,

            sqft_above,

            sqft_basement
        )
    )


    st.success(
        f'Estimated House Price: $ {prediction:,.2f}'
    )


    # VISUALIZATION

    st.subheader(
        'House Overview'
    )


    graph_df = pd.DataFrame({

        'Feature': [

            'Bedrooms',

            'Bathrooms',

            'Floors'

        ],

        'Value': [

            bedrooms,

            bathrooms,

            floors
        ]
    })


    fig, ax = plt.subplots()

    ax.bar(
        graph_df['Feature'],
        graph_df['Value']
    )

    ax.set_title(
        'House Features'
    )

    st.pyplot(fig)


st.markdown('---')

st.caption(
    'Built using Streamlit, Random Forest Regression, and Scikit-learn.'
)