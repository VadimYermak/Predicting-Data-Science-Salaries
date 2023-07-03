# Basic Parameters for Application

    # Prediction Model
        # Search Salaries by Country
        # By Education level
        # By Experience level

        # Based off of those parameters calculate salary
        # Compare median salary with predicted salary.

    # Exploratory Component 
        
        # page with graphs and data
        # Data will consist of imported data overview
        # Plot country data
        # Chart with mean salaries by country
        


""" 
Import libraries
    import pandas as pd
    import numpy as np
    import pickle #loads data
    import streamlit as st 
    import matplotlib.pyplot as plt
    import sklearn.preprocessing import LabelEncoder
    import sklearn.linear_model import LinearRegression
    import sklearn.tree import DecisionTreeRegressor
    import sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error

Add and Import data from

Clean data:
    No columns drop - keep Colums
        In second refactor decide to drop or keep columns.
        Drop columns: employee_residence, remote ratio

    
    Remove null values
        df = df[df["salary_in_usd"].notnull()]
    
    Drop na values
        dropna()
    
    Clean Columns data
    
        Keep only Full Time "employment_type" (TBD - team should discuss this implementation)
        Remove small "company_location" data points because model will be confused by data points that are small
            df["company_location"].value_counts()
            Create a function that will remove inefficient data

        Utilize a box plot to seen where the median data lies
        Based on box plot information keep Salary data that is closer to the median values

Train Data:
    Transform Data to ML readable data that the model can understand
        LabelEncoder()
        .fit_transform(df[string column name])
        .unique()

    Begin training model
        Split data between features., all other data, and label., salary data
            X = df.drop("salary_in_usd", axis=1)
            y = df["salary_in_usd"]
        
        Split to testing and training 

        Train data:
            LinearRegression()

        Check error in training
            error = np.sqrt(mean_squared_error(y, y_pred))

        If error value is too large try a new regression method
        ReTrain Data:
            DecisionTreeRegressor(random_state=0)
            .fit(X, y.values)

            or 

            RandomForestRegressor()
        
        If error value is still not close to a reasonable value, import and try all regression methods
            - can import and use GridSearchCV to get a regression method that will be more precise

            max_depth = [None, 2, 4, 6 ,8 10, 12]
            parameters = {"max_depth": max_depth}
            regressor = DecisionTreeRegressor(random_state=0)
            gs = GridSearchCV(regressor, parameters, scoring="neg_mean_squared_error")
            gs.fit(X, y.values)
            regressor = gs.best_estimator
            regressor.fit(X, y.values)
            y_pred = regessor.preict(X)
            error = np.sqrt(mean_squared_error(y, y_pred))
            print("${:,.02f}".format(error))

        Create a function that will grab user input and re-encode it for the regressor model

Create Stremlit site:


"""