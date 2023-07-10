import pandas as pd
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)

salary_df = pd.read_csv("Resources/ds_salaries.csv")


def analyze_regressors(X_test, X_train, y_test, y_train):
    # Define the regressors
    regressors = [
        ("Random Forest", RandomForestRegressor()),
        ("Gradient Boosting", GradientBoostingRegressor()),
        ("KNN", KNeighborsRegressor()),
        ("Decision Tree", DecisionTreeRegressor()),
        ("Linear Regression", LinearRegression()),
        ("Support Vector", SVR()),
        ("Gaussian Process", GaussianProcessRegressor()),
    ]

    # Create an empty DataFrame to store the metrics
    metrics_df = pd.DataFrame(columns=["Model", "MAE", "MSE", "RMSE", "R2 Score"])

    # Iterate over each regressor
    for reg_name, reg in regressors:
        steps = [("MinMax", StandardScaler()), ("Regressor", reg)]
        pipeline = Pipeline(steps)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Calculate the evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = metrics.r2_score(y_test, y_pred)

        new_row = {
            "Model": reg_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2,
        }

        metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)

    # Sort the metrics DataFrame by RMSE and R2 Score in ascending order
    p_sorted_metrics_df = metrics_df.sort_values(by=["MAE"], ascending=[True])

    return p_sorted_metrics_df


def importance_analysis(X_train, y_train):
    X = salary_df.drop(columns="salary_in_usd")
    rf_model = RandomForestRegressor(random_state=1)
    rf_model = rf_model.fit(X_train, y_train)
    importances = rf_model.feature_importances_
    sorted(zip(importances, X.columns), reverse=True)[:20]

    # Create a dataframe of the important features
    importances_df = pd.DataFrame(sorted(zip(importances, X.columns), reverse=True)[:])

    # Rename the columns
    importances_df = importances_df.rename(columns={0: "Importance", 1: "Feature"})

    # Set the index
    importances_df = importances_df.set_index("Feature")

    # Sort the dataframe by feature importance
    importances_df = importances_df.sort_values(by="Importance", ascending=False)

    return importances_df
