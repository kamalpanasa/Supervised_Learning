import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def load_data(file_path):

    df = pd.read_csv(file_path)

    return df


def preprocess_data(df):

    # Remove unnecessary column
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fix missing values properly
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # Encode target column
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # Features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Numerical columns
    numerical_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    # Categorical columns
    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Numerical pipeline
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical pipeline
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    # Combined preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )