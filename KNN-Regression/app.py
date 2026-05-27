import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocessing import (
    load_data,
    preprocess_data
)

from src.predict import (
    predict_price
)


# PAGE CONFIG

st.set_page_config(
    page_title='Laptop Price Prediction',
    page_icon='💻',
    layout='wide'
)


# LOAD DATA

df = load_data()

X, y, scaler = preprocess_data(df)


# HEADER

st.title('💻 Laptop Price Prediction')

st.write(
    '''
    Predict laptop prices using
    KNN Regression and laptop specifications.
    '''
)


# SIDEBAR

st.sidebar.header('Project Information')

st.sidebar.info(
    '''
    This application predicts laptop prices
    using K-Nearest Neighbors Regression.
    '''
)


# INPUT SECTION

st.subheader('Laptop Specifications')

col1, col2 = st.columns(2)


with col1:

    company = st.selectbox(
        'Company',
        [
            0, 1, 2, 3, 4, 5
        ]
    )

    typename = st.selectbox(
        'Laptop Type',
        [
            0, 1, 2, 3
        ]
    )

    inches = st.slider(
        'Screen Size (Inches)',
        10.0,
        20.0,
        15.6
    )

    ram = st.slider(
        'RAM (GB)',
        2,
        64,
        8
    )


with col2:

    memory = st.selectbox(
        'Storage Type',
        [
            0, 1, 2, 3
        ]
    )

    opsys = st.selectbox(
        'Operating System',
        [
            0, 1, 2, 3
        ]
    )

    weight = st.slider(
        'Weight (kg)',
        0.5,
        5.0,
        2.0
    )


st.markdown('---')


# PREDICTION

if st.button(
    'Predict Laptop Price',
    use_container_width=True
):

    prediction = predict_price(

        scaler,

        company,

        typename,

        inches,

        ram,

        memory,

        opsys,

        weight
    )


    st.success(
        f'Estimated Laptop Price: € {prediction:.2f}'
    )


    # VISUALIZATION

    st.subheader(
        'Laptop Specification Overview'
    )


    graph_df = pd.DataFrame({

        'Feature': [
            'RAM',
            'Weight',
            'Screen Size'
        ],

        'Value': [
            ram,
            weight,
            inches
        ]
    })


    fig, ax = plt.subplots()

    ax.bar(
        graph_df['Feature'],
        graph_df['Value']
    )

    ax.set_title(
        'Laptop Specifications'
    )

    st.pyplot(fig)


st.markdown('---')

st.caption(
    'Built using Streamlit, KNN Regression, and Scikit-learn.'
)