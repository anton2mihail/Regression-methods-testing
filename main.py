import pandas as pd
import math
from pandas import Series, DataFrame
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt


def prepare_data(df, forecast_col, forecast_out, test_size):
    # creating new column called label with the last 13 rows are nan
    label = df[forecast_col].shift(-forecast_out)
    X = np.array(df[[forecast_col]])  # creating the feature array
    true_data = X[-forecast_out:]
    X = preprocessing.scale(X)  # processing the feature array
    # creating the column i want to use later in the predicting method
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]  # X that will contain the training and testing
    label.dropna(inplace=True)  # dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=test_size)  # cross validation

    response = [X_train, X_test, Y_train, Y_test, X_lately, true_data]
    return response


data = pd.read_csv("^GSPC.csv", index_col=[
                   0], date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"))


# Adding two new features
data['HL_PCT'] = (data['High'] - data['Low']) / data['Close'] * 100.0
data['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0


forecast_col = 'Adj Close'  # choosing which column to forecast
forecast_out = 13  # how far to forecast
test_size = 0.5  # the size of my test set


# Data Preperation for training and testing
X_train, X_test, Y_train, Y_test, X_lately, true_data = prepare_data(
    data, forecast_col, forecast_out, test_size)

# initializing basic linear regression model
basic_lin_reg = linear_model.LinearRegression()

# training the linear regression model
basic_lin_reg.fit(X_train, Y_train)

# testing the linear regression model
score_basic_lin_reg = basic_lin_reg.score(
    X_test, Y_test)

# set that will contain the forecasted data
forecast_basic_lin_reg = basic_lin_reg.predict(X_lately)
forecast_basic_lin_reg = np.array(
    forecast_basic_lin_reg)  # changing to a np array

data['Lin Reg Forecast'] = np.nan

# Plot for Control data
adj_close = data["Adj Close"]
# Moving Average
mavg = adj_close.rolling(window=100).mean()
# Forecasted Data being inserted
last_date = data.iloc[-18].name
next_date = last_date
datapoints = len(forecast_basic_lin_reg)
idx = 0

for i in range(1, 18):
    if not datapoints == idx:
        if next_date.weekday() != 5 and next_date.weekday() != 6:
            data.loc[next_date] = forecast_basic_lin_reg[idx]
            idx += 1
        else:
            data.loc[next_date] = math.nan
        next_date += timedelta(days=1)

data['Adj Close'].tail(250).plot()
data['Lin Reg Forecast'].tail(250).plot()
mavg.tail(250).plot(label="MAVG")
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Lasso
lasso_lin_reg = linear_model.Lasso(alpha=0.1)
lasso_lin_reg.fit(X_train, Y_train)
score_lasso_lin_reg = lasso_lin_reg.score(X_test, Y_test)
forecast_lasso_lin_reg = lasso_lin_reg.predict(X_lately)


# Bayesian Ridge
bayesian_ridge_lin_reg = linear_model.BayesianRidge()
bayesian_ridge_lin_reg.fit(X_train, Y_train)
score_bayesian_ridge_lin_reg = bayesian_ridge_lin_reg.score(X_test, Y_test)
forecast_bayesian_ridge_lin_reg = bayesian_ridge_lin_reg.predict(X_lately)

# LARS Lasso
lars_lin_reg = linear_model.LassoLars(alpha=.1)
lars_lin_reg.fit(X_train, Y_train)
score_lars_lin_reg = lars_lin_reg.score(X_test, Y_test)
forecast_lars_lin_reg = lars_lin_reg.predict(X_lately)


#  This data frame will contain the results of all the different LIN REG methods
response = pd.DataFrame(
    columns=['True Values', 'Lasso', 'Bayesian Ridge', 'LARS Lasso'])
response["True Values"] = true_data.flatten()
response['Control'] = forecast_basic_lin_reg
response["Lasso"] = forecast_lasso_lin_reg
response["Bayesian Ridge"] = forecast_bayesian_ridge_lin_reg
response["Lars Lasso"] = forecast_lars_lin_reg
print(response)

# Display results
x_axis_vals = ["Basic Lin Reg", "Lasso", "Bayesian Ridge", 'LARS Lasso']
y_axis_vals = [score_basic_lin_reg, score_lasso_lin_reg,
               score_bayesian_ridge_lin_reg, score_lars_lin_reg]

# Probably not the best visual but here's a bar
plt.bar(x_axis_vals, y_axis_vals, align='center')
plt.title('Score Values')
plt.ylabel('Scores')
plt.xlabel('Methods')
plt.show()


# Show all Preds
x_true = response.index

plt.subplots()
