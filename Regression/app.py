import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.predict import predict_price
from src.preprocessing import load_data


# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    layout="wide"
)


# Load dataset

def get_data():

    df = load_data(
        "/data/raw/train.csv"
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
        "House Price Prediction"
    ]
)


# =====================================================
# PROJECT OVERVIEW
# =====================================================

if page == "Project Overview":

    st.title("House Price Prediction Dashboard")

    st.markdown("---")

    st.subheader("Project Objective")

    st.write(
        "This project predicts house prices "
        "using Random Forest Regressor."
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

    # Sale Price Distribution
    st.subheader("Sale Price Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.histplot(
        df["SalePrice"],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(
        include=["int64", "float64"]
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm"
    )

    st.pyplot(fig)

    # Overall Quality vs Sale Price
    st.subheader("Overall Quality vs Sale Price")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.boxplot(
        x=df["OverallQual"],
        y=df["SalePrice"],
        ax=ax
    )

    st.pyplot(fig)

    # Living Area vs Sale Price
    st.subheader("Living Area vs Sale Price")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.scatterplot(
        x=df["GrLivArea"],
        y=df["SalePrice"],
        ax=ax
    )

    st.pyplot(fig)


# =====================================================
# PREDICTION
# =====================================================

elif page == "House Price Prediction":

    st.title("House Price Prediction")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        overall_qual = st.slider(
            "Overall Quality",
            1,
            10,
            5
        )

        gr_liv_area = st.number_input(
            "Ground Living Area",
            min_value=0,
            value=1500
        )

        garage_cars = st.slider(
            "Garage Cars",
            0,
            5,
            2
        )

        garage_area = st.number_input(
            "Garage Area",
            min_value=0,
            value=500
        )

        total_bsmt_sf = st.number_input(
            "Total Basement Area",
            min_value=0,
            value=800
        )

    with col2:

        first_flr_sf = st.number_input(
            "First Floor Area",
            min_value=0,
            value=1000
        )

        full_bath = st.slider(
            "Full Bathrooms",
            0,
            5,
            2
        )

        total_rooms = st.slider(
            "Total Rooms Above Ground",
            1,
            15,
            6
        )

        year_built = st.slider(
            "Year Built",
            1900,
            2025,
            2000
        )

    if st.button("Predict House Price"):

        input_data = {

            "OverallQual": overall_qual,
            "GrLivArea": gr_liv_area,
            "GarageCars": garage_cars,
            "GarageArea": garage_area,
            "TotalBsmtSF": total_bsmt_sf,
            "1stFlrSF": first_flr_sf,
            "FullBath": full_bath,
            "TotRmsAbvGrd": total_rooms,
            "YearBuilt": year_built
        }

        prediction = predict_price(input_data)

        st.markdown("---")

        st.success(
            f"Predicted House Price: "
            f"${prediction:,.2f}"
        )
