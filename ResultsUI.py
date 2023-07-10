import streamlit as st
import pandas as pd


# Clean data method
def clean_categories(categories, cutoff):
    category_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            category_map[categories.index[i]] = categories.index[i]
        else:
            category_map[categories.index[i]] = "Other"
    return category_map


# @st.cache_data
def load_data():
    # Read data
    salary_df = pd.read_csv("Resources/ds_salaries.csv")

    # Clean data
    values = ["FT"]
    salary_df = salary_df[salary_df["employment_type"].isin(values)]
    salary_df = salary_df.drop(
        [
            "salary",
            "salary_currency",
            "employee_residence",
            "work_year",
            "employment_type",
        ],
        axis=1,
    )
    salary_df = salary_df[salary_df["salary_in_usd"].notnull()]
    salary_df = salary_df.dropna()

    # Clean company location column data
    company_location_map = clean_categories(
        salary_df["company_location"].value_counts(), 100
    )
    salary_df["company_location"] = salary_df["company_location"].map(
        company_location_map
    )
    salary_df = salary_df[salary_df["salary_in_usd"] <= 175000]
    salary_df = salary_df[salary_df["salary_in_usd"] >= 25000]

    # Clean job title column data
    job_title_map = clean_categories(salary_df["job_title"].value_counts(), 100)
    salary_df["job_title"] = salary_df["job_title"].map(job_title_map)
    return salary_df


salary_df = load_data()


# Show data results method for UI
def show_results_data(
    experience_level, job_title, remote_ratio, company_location, company_size
):
    df = salary_df.loc[
        (salary_df["experience_level"] == experience_level)
        & (salary_df["job_title"] == job_title)
        & (salary_df["remote_ratio"] == remote_ratio)
        & (salary_df["company_location"] == company_location)
        & (salary_df["company_size"] == company_size)
    ]
    st.dataframe(df)
