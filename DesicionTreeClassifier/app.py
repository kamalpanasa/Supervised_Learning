import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.predict import (
    predict_attrition
)


# PAGE CONFIG

st.set_page_config(
    page_title='Employee Attrition Prediction',
    page_icon='📊',
    layout='wide'
)


# HEADER

st.title(
    '📊 Employee Attrition Prediction'
)

st.write(
    '''
    Predict employee attrition using
    Decision Tree Classification.
    '''
)


# SIDEBAR

st.sidebar.header(
    'Project Information'
)

st.sidebar.info(
    '''
    This application predicts whether
    an employee is likely to leave
    the organization.
    '''
)


# INPUT SECTION

st.subheader(
    'Employee Information'
)

col1, col2 = st.columns(2)


with col1:

    age = st.slider(
        'Age',
        18,
        60,
        30
    )

    daily_rate = st.slider(
        'Daily Rate',
        100,
        1500,
        500
    )

    distance_from_home = st.slider(
        'Distance From Home',
        1,
        30,
        5
    )

    monthly_income = st.slider(
        'Monthly Income',
        1000,
        30000,
        10000
    )

    total_working_years = st.slider(
        'Total Working Years',
        0,
        40,
        5
    )

    years_at_company = st.slider(
        'Years At Company',
        0,
        40,
        3
    )


with col2:

    overtime = st.selectbox(
        'OverTime',
        [0, 1]
    )

    work_life_balance = st.selectbox(
        'Work Life Balance',
        [1, 2, 3, 4]
    )

    job_satisfaction = st.selectbox(
        'Job Satisfaction',
        [1, 2, 3, 4]
    )

    environment_satisfaction = st.selectbox(
        'Environment Satisfaction',
        [1, 2, 3, 4]
    )

    performance_rating = st.selectbox(
        'Performance Rating',
        [1, 2, 3, 4]
    )


st.markdown('---')


# PREDICTION

if st.button(
    'Predict Attrition',
    use_container_width=True
):

    prediction, probability = (
        predict_attrition(

            age,

            0,

            daily_rate,

            0,

            distance_from_home,

            2,

            0,

            environment_satisfaction,

            1,

            50,

            3,

            2,

            0,

            job_satisfaction,

            1,

            monthly_income,

            1,

            overtime,

            15,

            performance_rating,

            total_working_years,

            work_life_balance,

            years_at_company
        )
    )


    if prediction == 1:

        st.error(
            'Employee Likely to Leave'
        )

    else:

        st.success(
            'Employee Likely to Stay'
        )


    st.subheader(
        'Employee Overview'
    )


    graph_df = pd.DataFrame({

        'Feature': [

            'Income',

            'Experience',

            'Company Years'
        ],

        'Value': [

            monthly_income,

            total_working_years,

            years_at_company
        ]
    })


    fig, ax = plt.subplots()

    ax.bar(
        graph_df['Feature'],
        graph_df['Value']
    )

    ax.set_title(
        'Employee Metrics'
    )

    st.pyplot(fig)


st.markdown('---')

st.caption(
    'Built using Streamlit, Decision Tree Classification, and Scikit-learn.'
)