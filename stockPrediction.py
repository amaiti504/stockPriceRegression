import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
import numpy as np

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2019, 1, 11)

df = web.DataReader("MSFT", 'yahoo', start, end)
df = df[['Adj Close']]

# A variable for predicting 'n' days out into the future
forecast_out = 30

#Create another column (the target or dependent variable) shifted 'n' units up
df['Prediction'] = df[['Adj Close']].shift(-forecast_out)

# Convert the dataframe to a numpy array
X = np.array(df.drop(['Prediction'],1))

#Remove the last 'n' rows
X = X[:-forecast_out]

### Create the dependent data set (y)  #####
# Convert the dataframe to a numpy array (All of the values including the NaN's)
y = np.array(df['Prediction'])
# Get all of the y values except the last 'n' rows
y = y[:-forecast_out]

# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the Linear Regression  Model - 1
lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)
# Print linear regression model predictions for the test set
lr_prediction = lr.predict(x_test)
print(lr_prediction)
# Testing Model: R^2 for Linear Regression 
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)


# Quadratic Regression 2
qr2 = make_pipeline(PolynomialFeatures(2), Ridge())
qr2.fit(x_train, y_train)
# Print QUadratic regression model(2) predictions for the test set
qr2_prediction = qr2.predict(x_test)
print(qr2_prediction)
# Testing Model: R^2 for Quadratic Regression 2
qr2_confidence = qr2.score(x_test, y_test)
print("qr2 confidence: ", qr2_confidence)

# Quadratic Regression 3
qr3 = make_pipeline(PolynomialFeatures(3), Ridge())
qr3.fit(x_train, y_train)
# Print QUadratic regression model(3) predictions for the test set
qr3_prediction = qr3.predict(x_test)
print(qr3_prediction)
# Testing Model: R^2 for Quadratic Regression 3
qr3_confidence = qr3.score(x_test, y_test)
print("qr3 confidence: ", qr3_confidence)

# Create and train the Support Vector Machine (Regressor)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)
# Print support vector regressor model predictions for the test set
svm_prediction = svr_rbf.predict(x_test)
print(svm_prediction)
#SVM Regressor R^2
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)