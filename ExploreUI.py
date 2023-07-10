import streamlit as st
import pandas as pd
import numpy as np
from PredictUI import load_model
from RegressorAnalysis import analyze_regressors, importance_analysis
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px


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

model_data = load_model()

X_train = model_data["X_train"]
X_test = model_data["X_test"]
y_train = model_data["y_train"]
y_test = model_data["y_test"]
regressor = model_data["model"]
training_predictions = model_data["training_predictions"]
testing_predictions = model_data["testing_predictions"]
experience_level = model_data["experience_level"]
job_title = model_data["job_title"]
company_location = model_data["company_location"]
company_size = model_data["company_size"]

analyzed_regressors = analyze_regressors(X_test, X_train, y_test, y_train)


def show_exploration_page():
    st.title("Explore Data Science Salaries Data")
    st.write("""### Kaggle Data Science Salaries 2023""")
    st.divider()

    # Plot full dataset set
    st.write("""#### Data Set""")
    st.dataframe(salary_df)

    # Data cleaning process and pre analysis
    st.write("""##### Data Cleaning""")
    fig = px.box(
        salary_df,
        x="company_location",
        y="salary_in_usd",
        title="Salary to Company Location",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.code(
        """Based on the dataset it was indicative to acheieve high precision in terms 
        of Salary.  These results allow for the model to create precieved 
        predictions with a salary range no higher than 200k and no lower than 25K.""",
        language="python",
    )

    # analyzed_regressor data set
    st.write("""##### Multi Regressor Error Analysis""")
    st.dataframe(analyzed_regressors)

    # Plot Job Title Data
    job_title_data = salary_df["job_title"].value_counts()
    colors = plt.get_cmap("Blues")(np.linspace(0.2, 0.7, len(job_title_data)))

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        job_title_data,
        colors=colors,
        radius=3,
        center=(4, 4),
        autopct="%1.1f%%",
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.axis("equal")
    ax.legend(
        wedges,
        job_title_data.index,
        title="Ingredients",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )

    st.write("""##### Data Science Job Titles""")
    st.pyplot(fig)

    st.write("""##### Prediction Analysis""")

    st.write("Coefficients: \n", regressor.coef_)
    st.write(
        "Mean squared error: %.2f" % mean_squared_error(y_test, testing_predictions)
    )
    st.write(
        "Coefficient of determination: %.2f" % r2_score(y_test, testing_predictions)
    )

    st.divider()

    # Early unclean data analysis
    importances_df = importance_analysis(X_train, y_train)
    st.write("""##### Uncleaned Data Analysis""")
    st.dataframe(importances_df)

    st.write("""###### Feature importance""")
    st.code(
        """ 
        Feature importance is a measure of the relative importance or relevance of 
        each feature to the model's prediction. The number in the chart below 
        quantifies how much each feature contributes to the model's overall performance 
        in making accurate predictions. We used this information in decision to 
        identify influential features, and gain further insights into the model's 
        behavior.
        """,
        language="python",
    )

    st.write("""###### Correlation to the Target Variable""")
    st.code(
        """ 
        We used this metric to assess the relationship between each individual 
        feature and the target variable (salary_in_usd). A positive correlation 
        means that as the feature increases, the target variable tends to increase, 
        while negative correlation means that as the feature increases, the 
        target variable tends to decrease. Correlation to the target variable was 
        also to gain insights in the feature's behavior.
        """,
        language="python",
    )

    st.divider()
    st.caption(
        "Here and below is where future interations of new implentations features will live! :sunglasses:"
    )
