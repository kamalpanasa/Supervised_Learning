import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.predict import (
    predict_attack
)


st.set_page_config(
    page_title='Network Intrusion Detection',
    page_icon='🛡️',
    layout='wide'
)


st.title(
    '🛡️ Network Intrusion Detection'
)

st.write(
    '''
    Predict whether network traffic
    is normal or anomalous using
    Random Forest Classification.
    '''
)


st.sidebar.header(
    'Project Information'
)

st.sidebar.info(
    '''
    Random Forest based
    Network Intrusion Detection System.
    '''
)


st.subheader(
    'Network Traffic Features'
)


col1, col2 = st.columns(2)


with col1:

    duration = st.number_input(
        'Duration',
        value=0
    )

    src_bytes = st.number_input(
        'Source Bytes',
        value=0
    )

    dst_bytes = st.number_input(
        'Destination Bytes',
        value=0
    )

    count = st.number_input(
        'Connection Count',
        value=0
    )

    srv_count = st.number_input(
        'Service Count',
        value=0
    )


with col2:

    protocol_type = st.selectbox(
        'Protocol Type',
        [0, 1, 2]
    )

    service = st.number_input(
        'Service',
        value=0
    )

    flag = st.number_input(
        'Flag',
        value=0
    )

    same_srv_rate = st.slider(
        'Same Service Rate',
        0.0,
        1.0,
        0.5
    )

    diff_srv_rate = st.slider(
        'Different Service Rate',
        0.0,
        1.0,
        0.1
    )


if st.button(
    'Predict Traffic'
):

    features = [

        duration,
        protocol_type,
        service,
        flag,
        src_bytes,
        dst_bytes,

        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

        count,
        srv_count,

        0.0,
        0.0,
        0.0,
        0.0,

        same_srv_rate,
        diff_srv_rate,

        0.0,
        0,
        0,

        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ]


    prediction, probability = (
        predict_attack(
            features
        )
    )


    if prediction == 1:

        st.error(
            'Anomalous Traffic Detected'
        )

    else:

        st.success(
            'Normal Traffic Detected'
        )


    anomaly_probability = (
        probability[1] * 100
    )


    st.subheader(
        'Anomaly Probability'
    )

    st.progress(
        anomaly_probability / 100
    )

    st.write(
        f'{anomaly_probability:.2f}% Risk'
    )


    graph_df = pd.DataFrame({

        'Feature': [

            'Source Bytes',

            'Destination Bytes',

            'Connections'
        ],

        'Value': [

            src_bytes,

            dst_bytes,

            count
        ]
    })


    fig, ax = plt.subplots()

    ax.bar(
        graph_df['Feature'],
        graph_df['Value']
    )

    ax.set_title(
        'Traffic Analysis'
    )

    st.pyplot(fig)