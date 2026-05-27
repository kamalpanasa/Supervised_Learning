import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.predict import (
    predict_insurance_cost
)


# PAGE CONFIG

st.set_page_config(
    page_title='Insurance Cost Prediction',
    page_icon='💰',
    layout='wide'
)


# HEADER

st.title(
    '💰 Insurance Cost Prediction'
)

st.write(
    '''
    Predict medical insurance charges
    using Decision Tree Regression.
    '''
)


# SIDEBAR

st.sidebar.header(
    'Project Information'
)

st.sidebar.info(
    '''
    This application predicts insurance costs
    based on health and demographic information.
    '''
)


# INPUT SECTION

st.subheader(
    'Customer Information'
)

col1, col2 = st.columns(2)


with col1:

    age = st.slider(
        'Age',
        18,
        100,
        30
    )

    bmi = st.slider(
        'BMI',
        10.0,
        50.0,
        25.0
    )

    children = st.slider(
        'Children',
        0,
        10,
        1
    )


with col2:

    sex = st.selectbox(
        'Sex',
        [0, 1]
    )

    smoker = st.selectbox(
        'Smoker',
        [0, 1]
    )

    region = st.selectbox(
        'Region',
        [0, 1, 2, 3]
    )


st.markdown('---')


# PREDICTION

if st.button(
    'Predict Insurance Charges',
    use_container_width=True
):

    prediction = (
        predict_insurance_cost(

            age,

            sex,

            bmi,

            children,

            smoker,

            region
        )
    )


    st.success(
        f'Estimated Insurance Charges: $ {prediction:.2f}'
    )


    # VISUALIZATION

    st.subheader(
        'Customer Health Overview'
    )


    graph_df = pd.DataFrame({

        'Feature': [

            'Age',

            'BMI',

            'Children'
        ],

        'Value': [

            age,

            bmi,

            children
        ]
    })


    fig, ax = plt.subplots()

    ax.bar(
        graph_df['Feature'],
        graph_df['Value']
    )

    ax.set_title(
        'Customer Metrics'
    )

    st.pyplot(fig)


    # INSIGHTS

    st.subheader(
        'Health Insights'
    )

    if smoker == 1:
        st.warning(
            'Smoking significantly increases insurance costs.'
        )

    if bmi > 30:
        st.warning(
            'Higher BMI may increase insurance charges.'
        )

    if age > 50:
        st.info(
            'Older age groups generally have higher insurance costs.'
        )


st.markdown('---')

st.caption(
    'Built using Streamlit, Decision Tree Regression, and Scikit-learn.'
)