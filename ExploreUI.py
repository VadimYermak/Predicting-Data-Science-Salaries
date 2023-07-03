import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def clean_categories(categories, cutoff):
    category_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            category_map[categories.index[i]] = categories.index[i]
        else:
            category_map[categories.index[i]] = 'Other'
    return category_map

# @st.cache_data
def load_data():
    # Read data
    salary_df = pd.read_csv("Resources/ds_salaries.csv")

    # Clean data
    salary_df = salary_df.drop(['salary', 'salary_currency', 'employee_residence'], axis=1)
    salary_df = salary_df[salary_df["salary_in_usd"].notnull()]
    salary_df = salary_df.dropna()

    values = ['FT']
    salary_df = salary_df[salary_df['employment_type'].isin(values)]

    # Clean company location column data
    company_location_map = clean_categories(salary_df['company_location'].value_counts(), 100)
    salary_df['company_location'] = salary_df['company_location'].map(company_location_map)
    salary_df = salary_df[salary_df['salary_in_usd'] <= 175000]
    salary_df = salary_df[salary_df['salary_in_usd'] >= 25000]

    # Clean job title column data
    job_title_map = clean_categories(salary_df['job_title'].value_counts(), 100)
    salary_df['job_title'] = salary_df['job_title'].map(job_title_map)
    return salary_df

salary_df = load_data()

def show_exploration_page():
    st.title("Explore Data Science Salaries Data")
    st.write("""### Kaggle Data Science Salaries 2023""")

    # Plot Job Title Data
    job_title_data = salary_df['job_title'].value_counts()
    colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(job_title_data)))

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(job_title_data, colors=colors, radius=3, center=(4, 4), autopct="%1.1f%%", wedgeprops={"linewidth": 1, "edgecolor": "white"})
    ax.axis('equal')
    ax.legend(wedges, job_title_data.index,
          title="Ingredients",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
    
    st.write("""#### Data Science Job Titles""")
    st.pyplot(fig)