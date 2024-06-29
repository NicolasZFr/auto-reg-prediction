# Automobile business analysis and supervised machine learning algorithms (price prediction)

> :loudspeaker: Algorithm descriptions were taken from ChatGPT and summarized

## 1. Introduction

In the competitive automotive market, accurately predicting the price of a vehicle is essential for both manufacturers and consumers. This project focuses on developing a predictive model that is based exclusively on a key variable: engine size.

By simplifying the analysis and focusing on this fundamental characteristic, we can explore how it directly influences the price of the vehicle, without being affected by other factors that could complicate the analysis. Using machine learning techniques, our goal is to build a robust model that provides accurate estimates of a vehicle's price based on its engine size, making it easier for manufacturers and consumers to make more informed decisions in the automotive market.

## 2. Data Collection and Preprocessing

Importing necessary libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
```
Importing table with column names
```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
headers = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]

pd.read_csv(url, header = None)
df.columns = headers
```
### Dropping missing values
We will drop the NaN values in the price column since we need it with complete and real data because we’re gonna use it to predict prices based in data
```python
df["price"] = pd.to_numeric(df["price"], errors='coerce')
df.dropna(subset=["price"], axis=0, inplace=True)
```
### Replacing missing values
In the case of the others, we replace the missing value with the average
```python
df.replace("?", np.nan, inplace = True)
df["normalized-losses"] = pd.to_numeric(df["normalized-losses"], errors='coerce')
mean = df["normalized-losses"].mean()
df["normalized-losses"].replace(np.nan, mean, inplace = True)

avg_norm_loss = df["bore"].astype("float").mean(axis=0)
df["bore"].replace(np.nan, avg_norm_loss, inplace=True)

avg_norm_loss = df["stroke"].astype("float").mean(axis=0)
df["stroke"].replace(np.nan, avg_norm_loss, inplace=True)

avg_norm_loss = df["horsepower"].astype("float").mean(axis=0)
df["horsepower"].replace(np.nan, avg_norm_loss, inplace=True)
df["horsepower"] = df["horsepower"].astype("float")

avg_norm_loss = df["peak-rpm"].astype("float").mean(axis=0)
df["peak-rpm"].replace(np.nan, avg_norm_loss, inplace=True)
```
Replacing the NaN cells in num-doors with the mode
```python
df["num-of-doors"].replace(np.nan, "four", inplace=True)
```
### Data formatting
The measures are in English system, so we’ll adjust them to universal ones
```python
df["city-mpg"] = 235/df["city-mpg"] # adjusting the formula
df.rename(columns={"city-mpg": "city-L/100km"}, inplace = True) # from "miles per galon" to "liter per 100km"
```
### Changing from objects to numbers
```python
df.rename(columns={"fuel-type":"fuel"},inplace=True)

fuel_dummies = pd.get_dummies(df["fuel"]).astype("int")
df = pd.concat([df, fuel_dummies], axis=1)
```
## 3.Exploratory Data Analysis (EDA)
### Price trend based on drive wheels (with box plot)
We import seaborn and make a graph with “drive-wheels” in the X axis and “price” in the y axis
<div style="width:100%;display:flex;justify-content:center">
    <table>
        <thead>
            <th>Model</th>
            <th>RMSE</th>
            <th>R^2</th>
        </thead>
        <tbody>
            <tr>
                <td>Polynomial regressor</td>
                <td>3718</td>
                <td>0.5996</td>
            </tr>
            <tr>
                <td>Decission Tree regressor</td>
                <td>5138</td>
                <td>0.2354</td>
            </tr>
            <tr>
                <td>SVM regressor</td>
                <td>3909</td>
                <td>0.5574</td>
            </tr>
            <tr>
                <td>K-neighbors regressor</td>
                <td>3465</td>
                <td>0.6523</td>
            </tr>
            <tr>
                <td>XGBoost regressor</td>
                <td>2610</td>
                <td>0.8026</td>
            </tr>
            <tr>
                <td>Voting regressor</td>
                <td>3327</td>
                <td>0.6794</td>
            </tr>
        </tbody>
    </table>
</div>