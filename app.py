import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.predict import predict_transaction

# PAGE CONFIG
st.set_page_config(
    page_title='Credit Card Fraud Detection',
    page_icon='💳',
    layout='wide'
)

# HEADER
st.title(
    '💳 Credit Card Fraud Detection'
)

st.write(
    '''
    Detect fraudulent credit card transactions
    using Random Forest Classification.
    '''
)

# SIDEBAR

st.sidebar.header(
    'Project Information'
)

st.sidebar.info(
    '''
    This application predicts whether
    a transaction is fraudulent or legitimate.
    '''
)


# INPUT SECTION

st.subheader(
    'Transaction Details'
)


amount = st.slider(
    'Transaction Amount',
    0.0,
    5000.0,
    100.0
)


time = st.slider(
    'Transaction Time',
    0,
    200000,
    50000
)

st.write(
    '### PCA Features'
)

pca_features = []
cols = st.columns(4)

for i in range(1, 29):
    with cols[(i - 1) % 4]:
        value = st.number_input(
            f'V{i}',
            value=0.0,
            format='%.4f'
        )
        pca_features.append(value)

st.markdown('---')


# PREDICTION

if st.button(
    'Detect Fraud',
    use_container_width=True
):

    transaction_data = [

        time,

        *pca_features,

        amount
    ]


    prediction, probability = (
        predict_transaction(
            transaction_data
        )
    )


    if prediction == 1:

        st.error(
            'Fraudulent Transaction Detected'
        )

    else:

        st.success(
            'Legitimate Transaction'
        )


    fraud_probability = (probability[1] * 100)
    st.subheader('Fraud Probability')
    st.progress(fraud_probability / 100)
    st.write(f'{fraud_probability:.2f}% Fraud Risk')

    # VISUALIZATION

    st.subheader('Transaction Overview')

    graph_df = pd.DataFrame({
        'Feature': [
            'Amount',
            'Time'
        ],
        'Value': [
            amount,
            time
        ]
    })

    fig, ax = plt.subplots()

    ax.bar(
        graph_df['Feature'],
        graph_df['Value']
    )

    ax.set_title(
        'Transaction Metrics'
    )

    st.pyplot(fig)


st.markdown('---')

st.caption(
    'Built using Streamlit, Random Forest Classification, and Scikit-learn.'
)