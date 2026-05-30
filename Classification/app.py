import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.predict import predict_churn
from src.preprocessing import load_data


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide"
)


# -----------------------------
# Load Dataset
# -----------------------------

@st.cache_data

def get_data():

    df = load_data(
        "data/raw/Telco-Customer-Churn.csv"
    )

    return df


# -----------------------------
# Load Data
# -----------------------------


df = get_data()


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Project Overview",
        "Dataset Overview",
        "Visualizations",
        "Churn Prediction"
    ]
)


# =========================================================
# PROJECT OVERVIEW
# =========================================================

if page == "Project Overview":

    st.title("Customer Churn Prediction Dashboard")

    st.markdown("---")

    st.subheader("Project Objective")

    st.write(
        "This project predicts whether a customer is likely to churn "
        "using the AdaBoost Classification algorithm."
    )

    st.subheader("Technologies Used")

    st.write(
        "- Python\n"
        "- Scikit-learn\n"
        "- Streamlit\n"
        "- Pandas\n"
        "- Matplotlib\n"
        "- Seaborn"
    )

    st.subheader("Dataset Information")

    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")
# =========================================================
# DATASET OVERVIEW
# =========================================================

elif page == "Dataset Overview":

    st.title("Dataset Overview")

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Column Names")

    st.write(df.columns.tolist())

    st.subheader("Missing Values")

    missing_values = df.isnull().sum()

    st.dataframe(missing_values)

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

# =========================================================
# VISUALIZATIONS
# =========================================================

elif page == "Visualizations":

    st.title("Data Visualizations")

    st.markdown("---")


    # -----------------------------
    # Churn Distribution
    # -----------------------------

    st.subheader("Churn Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(
        x="Churn",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

  # -----------------------------
    # Gender Distribution
    # -----------------------------

    st.subheader("Gender Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(
        x="gender",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

      # -----------------------------
    # Contract Type Distribution
    # -----------------------------

    st.subheader("Contract Type Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.countplot(
        x="Contract",
        data=df,
        ax=ax
    )

    plt.xticks(rotation=10)

    st.pyplot(fig)

    # -----------------------------
    # Internet Service Distribution
    # -----------------------------

    st.subheader("Internet Service Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.countplot(
        x="InternetService",
        data=df,
        ax=ax
    )

    st.pyplot(fig)


    # -----------------------------
    # Internet Service Distribution
    # -----------------------------

    st.subheader("Internet Service Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.countplot(
        x="InternetService",
        data=df,
        ax=ax
    )

    st.pyplot(fig)


    # -----------------------------
    # Tenure Distribution
    # -----------------------------

    st.subheader("Tenure Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.histplot(
        df["tenure"],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

  # -----------------------------
    # Churn vs Contract
    # -----------------------------

    st.subheader("Churn vs Contract")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.countplot(
        x="Contract",
        hue="Churn",
        data=df,
        ax=ax
    )

    st.pyplot(fig)
    # -----------------------------
    # Correlation Heatmap
    # -----------------------------

    st.subheader("Correlation Heatmap")

    temp_df = df.copy()

    temp_df["TotalCharges"] = pd.to_numeric(
        temp_df["TotalCharges"],
        errors="coerce"
    )

    numeric_df = temp_df.select_dtypes(
        include=["int64", "float64"]
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)
# =========================================================
# CHURN PREDICTION
# =========================================================

elif page == "Churn Prediction":

    st.title("Customer Churn Prediction")

    st.markdown("---")

    col1, col2 = st.columns(2)


    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )
        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure",
            0,
            72,
            12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    with col2:

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )
        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )
        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=50.0
        )
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=500.0
        )


    if st.button("Predict Churn"):

        input_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        prediction, probability = predict_churn(
            input_data
        )

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                f"Customer is likely to churn.\n\n"
                f"Probability: {probability:.2f}"
            )

        else:

            st.success(
                f"Customer is not likely to churn.\n\n"
                f"Probability: {probability:.2f}"
            )

        st.progress(float(probability))