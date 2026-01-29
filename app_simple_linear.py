import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config("ML Playground", layout="centered")

st.title("Machine Learning Playground")
st.write("Regression & Classification Models in One App")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Problem Type Selection
# --------------------------------------------------
problem_type = st.selectbox(
    "Select Problem Type",
    ["Regression", "Classification"]
)

# --------------------------------------------------
# Feature / Target
# --------------------------------------------------
X = df[["total_bill"]]

if problem_type == "Regression":
    y = df["tip"]
else:
    y = (df["tip"] > df["tip"].median()).astype(int)  # Binary classification

# --------------------------------------------------
# Model Selection
# --------------------------------------------------
if problem_type == "Regression":
    model_name = st.selectbox(
        "Select Regression Model",
        [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "ElasticNet",
            "Decision Tree",
            "Random Forest",
            "SVR",
            "KNN Regressor"
        ]
    )
else:
    model_name = st.selectbox(
        "Select Classification Model",
        [
            "Logistic Regression",
            "KNN Classifier",
            "Decision Tree",
            "Random Forest",
            "SVC",
            "Naive Bayes"
        ]
    )

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# Model Initialization
# --------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "ElasticNet": ElasticNet(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "SVR": SVR(),
    "KNN Regressor": KNeighborsRegressor(),

    "Logistic Regression": LogisticRegression(),
    "KNN Classifier": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVC": SVC(),
    "Naive Bayes": GaussianNB()
}

model = models[model_name]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
st.subheader("Model Performance")

if problem_type == "Regression":
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    c1, c2 = st.columns(2)
    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("RMSE", f"{rmse:.2f}")
    st.metric("R² Score", f"{r2:.3f}")

else:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc:.2f}")
    c2.metric("Precision", f"{prec:.2f}")
    c3, c4 = st.columns(2)
    c3.metric("Recall", f"{rec:.2f}")
    c4.metric("F1 Score", f"{f1:.2f}")

# --------------------------------------------------
# Visualization
# --------------------------------------------------
st.subheader("Visualization")

fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)

if problem_type == "Regression":
    X_all = scaler.transform(X)
    ax.plot(df["total_bill"], model.predict(X_all), color="red")

ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
st.pyplot(fig)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.subheader("Make a Prediction")

bill = st.slider(
    "Total Bill",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

bill_scaled = scaler.transform([[bill]])
prediction = model.predict(bill_scaled)[0]

if problem_type == "Regression":
    st.success(f"Predicted Tip: ${prediction:.2f}")
else:
    st.success("High Tip" if prediction == 1 else "Low Tip")
