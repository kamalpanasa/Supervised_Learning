from pathlib import Path

import numpy as np

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent


model_path = (
    BASE_DIR /
    'models' /
    'decision_tree_model.pkl'
)


model = joblib.load(
    model_path
)


def predict_attrition(

    age,

    business_travel,

    daily_rate,

    department,

    distance_from_home,

    education,

    education_field,

    environment_satisfaction,

    gender,

    hourly_rate,

    job_involvement,

    job_level,

    job_role,

    job_satisfaction,

    marital_status,

    monthly_income,

    num_companies_worked,

    overtime,

    percent_salary_hike,

    performance_rating,

    total_working_years,

    work_life_balance,

    years_at_company
):

    data = np.array([[

        age,

        business_travel,

        daily_rate,

        department,

        distance_from_home,

        education,

        education_field,

        environment_satisfaction,

        gender,

        hourly_rate,

        job_involvement,

        job_level,

        job_role,

        job_satisfaction,

        marital_status,

        monthly_income,

        num_companies_worked,

        overtime,

        percent_salary_hike,

        performance_rating,

        total_working_years,

        work_life_balance,

        years_at_company

    ]])


    prediction = model.predict(
        data
    )


    probability = model.predict_proba(
        data
    )


    return prediction[0], probability[0]