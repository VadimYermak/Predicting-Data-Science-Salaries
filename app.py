import streamlit as st
from PredictUI import show_prediction_page
from ExploreUI import show_exploration_page

# Title for tab bar
page = st.sidebar.selectbox(
    "Data Exploartion & Salary Prediction", ("Explore Data", "Predict Salary")
)

# Create Page selection tab
if page == "Explore Data":
    show_exploration_page()
else:
    show_prediction_page()
