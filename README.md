<h1 align="center">Predicting Data Science Salaries</h1>

## Table of contents

- [Description](#description)
- [Background](#background)
- [Feature Description](#feature-description)
- [Installation Requirements / Technologies](#installation-requirements-technologies)
- [Images](#images)
- [Collaborators](#collaborators)
- [Sources](#sources)
- [License](#license)

## Description
This project focuses on developing a machine learning model for predicting salaries. The objective is to develop a machine learning model that  estimates salaries based on various key features. For data cleaning, features were evaluated for importance and predictive value. Supervised learning and neural network models were developed and tested to deliver reliable predictions for salary-related decision-making.

## Background
Data science is a field that combines various techniques, algorithms, and tools to extract meaningful insights and knowledge from structured and unstructured data. Data science salaries can vary widely depending on several factors such as location, level of experience, industry, company size, and education. The rapid growth of the data science field has resulted in an increased demand for accurate salary predictions within the industry.

## Feature Description
This section provides a description of the features included in the dataset.
```
work_year: The year the salary was paid.
experience_level: The experience level in the job during the year
employment_type: The type of employment for the role
job_title: The role worked in during the year.
salary: The total gross salary amount paid.
salary_currency: The currency of the salary paid as an ISO 4217 currency code.
salary_in_usd: The salary in USD
employee_residence: Employee's primary country of residence in during the work year as an ISO 3166 country code.
remote_ratio: The overall amount of work done remotely
company_location: The country of the employer's main office or contracting branch
company_size: The median number of people that worked for the company during the year
```

## Installation Requirements
Import the following packages before running the script. Note that the user will need to create a .env file with their own API keys.
```
import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import fire
import questionary
import json
from MCForecastTools import MCSimulation

%matplotlib inline
```
## Images
The script allows user prompts to select from a list with very few manual entries.

| Select stocks from a list| Monte Carlo Simulation |
| :-----------------:| :-----------------:|
| <img src = "https://github.com/VadimYermak/Project-1/blob/main/Images/Stocks%20List.png" width="300">| <img src = "https://github.com/VadimYermak/Project-1/blob/main/Images/Monte%20Carlo%20Simulation.png" width="420"> |

User can also generate a Favorites List for their stocks for future reference.

<img src = "https://github.com/VadimYermak/Project-1/blob/main/Images/Favorites%20List.png" width="300">

## Collaborators
+ Philippe Timothe (PhilippeTimothe@gmail.com)
+ Bryan Lu (btl245@stern.nyu.edu)
+ Vadim Yermak (vadik.ermak@gmail.com)
+ John Nguyen (nguyenjohn1337@gmail.com)
+ Cherryl Adzang (cherryl.adzang@gmail.com)

## Sources
Columbia Engineering Bootcamp learning materials

[RANDOMARNAB]. [2023].[Data Science Salaries 2023 : Salaries of Different Data Science Fields in the Data Science Domain]. Retrieved [2023-06-23] from [https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023].

## License

    MIT License

    Copyright (c) 2023 Columbia University - CU-VIRT-FIN-PT-03-2023-U-B-TTH - Team 3

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
