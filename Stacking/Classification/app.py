import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.predict import predict_loan_status
from src.preprocessing import load_data


# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="wide"
)


# Load dataset
@st.cache_data
def get_data():

    df = load_data(
        "data/raw/train.csv"
    )

    return df


df = get_data()


# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Project Overview",
        "Dataset Overview",
        "Visualizations",
        "Loan Prediction"
    ]
)


# =====================================================
# PROJECT OVERVIEW
# =====================================================

if page == "Project Overview":

    st.title("Loan Approval Prediction Dashboard")

    st.markdown("---")

    st.subheader("Project Objective")

    st.write(
        "This project predicts loan approval "
        "using a Stacking Classifier Ensemble Model."
    )

    st.subheader("Base Models")

    st.write(
        "- Logistic Regression\n"
        "- Decision Tree Classifier\n"
        "- Random Forest Classifier"
    )

    st.subheader("Meta Model")

    st.write(
        "- Logistic Regression"
    )

    st.subheader("Dataset Information")

    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")


# =====================================================
# DATASET OVERVIEW
# =====================================================

elif page == "Dataset Overview":

    st.title("Dataset Overview")

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum())


# =====================================================
# VISUALIZATIONS
# =====================================================

elif page == "Visualizations":

    st.title("Data Visualizations")

    st.markdown("---")

    # Loan Status Distribution
    st.subheader("Loan Status Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(
        x="Loan_Status",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

    # Gender Distribution
    st.subheader("Gender Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(
        x="Gender",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

    # Education Distribution
    st.subheader("Education Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(
        x="Education",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

    # Applicant Income Distribution
    st.subheader("Applicant Income Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.histplot(
        df["ApplicantIncome"],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    # Loan Amount Distribution
    st.subheader("Loan Amount Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.histplot(
        df["LoanAmount"],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(
        include=["int64", "float64"]
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm"
    )

    st.pyplot(fig)


# =====================================================
# PREDICTION
# =====================================================

elif page == "Loan Prediction":

    st.title("Loan Approval Prediction")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
        )

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0,
            value=5000
        )

    with col2:

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0,
            value=2000
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            value=120
        )

        loan_term = st.number_input(
            "Loan Amount Term",
            min_value=0,
            value=360
        )

        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

    if st.button("Predict Loan Status"):

        input_data = {

            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area
        }

        prediction, probability = predict_loan_status(
            input_data
        )

        st.markdown("---")

        if prediction == 1:

            st.success(
                f"Loan Approved\n\n"
                f"Probability: {probability:.2f}"
            )

        else:

            st.error(
                f"Loan Rejected\n\n"
                f"Probability: {probability:.2f}"
            )

        st.progress(float(probability))