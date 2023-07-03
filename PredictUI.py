import numpy as np
import pickle
import streamlit as st


def load_model():
    with open('Resources/salaries_pred_model.pkl', 'rb') as file:
        model_data = pickle.load(file)
    return model_data

model_data = load_model()

regressor = model_data["model"]
experience_level = model_data["experience_level"]
employment_type = model_data["employment_type"]
job_title = model_data["job_title"]
company_location = model_data["company_location"]
company_size = model_data["company_size"]

def show_prediction_page():
    st.title("Data Science Salary Predictions")

    st.write("""### Input selection needed to predict salary""")

    experience_levels = (
        "EX",
        "EN",
        "MI",
        "SE"
    )    

    employment_types = (
        "FT",
    )

    job_titles = (
        "Data Engineer",
        "Data Scientist",
        "Data Analyst",
        "Machine Learning Engineer"
    )

    remote_ratios = (
        100,
        0
    )

    company_locations = (
        "US",
        "GB"
    )

    company_sizes = (
        "S",
        "M",
        "L"
    )

    work_year = st.slider("Work Year", 2020, 2023, 2023)
    experience_l = st.selectbox("Experience Level", experience_levels)
    employment_t = st.selectbox("Employment Type",employment_types)
    job_t = st.selectbox("Job Title", job_titles)
    remote = st.selectbox("Remote Percentage", remote_ratios)
    company_l = st.selectbox("Company Location", company_locations)
    company_s = st.selectbox("Company Size", company_sizes)

    calculate = st.button("Caluculate Salary Prediction")
    if calculate == True:
        X = np.array([[work_year, experience_l, employment_t, job_t, remote, company_l, company_s]])
        X[:, 1] = experience_level.transform(X[:, 1])
        X[:, 2] = employment_type.transform(X[:, 2])
        X[:, 3] = job_title.transform(X[:, 3])
        X[:, 5] = company_location.transform(X[:, 5])
        X[:, 6] = company_size.transform(X[:, 6])
        X = X.astype('int64')

        salary = regressor.predict(X)
        st.subheader(f"Predicted salary is ${salary[0]:.2f}")