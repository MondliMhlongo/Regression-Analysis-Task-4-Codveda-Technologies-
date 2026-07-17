import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error

dfStock = pd.read_csv("C:\\Data Set For Task\\2) Stock Prices Data Set.csv")
dfStock.columns = dfStock.columns.str.strip()
dfStock = dfStock.dropna()

dfIris = pd.read_csv("C:\\Data Set For Task\\1) iris.csv")
dfIris.columns = dfIris.columns.str.strip()
dfIris = dfIris.dropna()

dfSentiments = pd.read_csv("C:\\Data Set For Task\\3) Sentiment dataset.csv")
dfSentiments.columns = dfSentiments.columns.str.strip()
dfSentiments = dfSentiments.dropna()

print("Stock Prices: ", dfStock.head())
print("Iris In Metres: ", dfIris.head())
print("Sentiments: ", dfSentiments.head())

#x and y variables for each dataset
xStock = dfStock[["open", "close", "high", "low"]]
yStock = dfStock["volume"]

xIris = dfIris[["sepal_length", "sepal_width", "petal_width"]]
yIris = dfIris["petal_length"]

xSentiments = dfSentiments[["Likes"]]
ySentiments = dfSentiments["Hour"]

#Spliting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(xStock, yStock, test_size=0.2, random_state=42)
X_trainIris, X_testIris, y_trainIris, y_testIris = train_test_split(xIris, yIris, test_size=0.2, random_state=42)
X_trainSentiments, X_testSentiments, y_trainSentiments, y_testSentiments = train_test_split(xSentiments, ySentiments, test_size=0.2, random_state=42)

#training the MLR model on the training set
model = LinearRegression()
model.fit(X_train, y_train)

modelIris = LinearRegression()
modelIris.fit(X_trainIris, y_trainIris)

modelSentiments = LinearRegression()
modelSentiments.fit(X_trainSentiments, y_trainSentiments)

#Predicting the test set results
y_predict = model.predict(X_test)
y_predictIris = modelIris.predict(X_testIris)
y_predictSentiments = modelSentiments.predict(X_testSentiments)

#R-squared Errors
r2MetricIris = r2_score(y_testIris, y_predictIris)
r2ModelScoreIris = modelIris.score(X_testIris, y_testIris)

r2MetricStock = r2_score(y_test, y_predict)
r2ModelScoreStock = model.score(X_test, y_test)

r2MetricSentiments = r2_score(y_testSentiments, y_predictSentiments)
r2ModelScoreSentiments = modelSentiments.score(X_testSentiments, y_testSentiments)

#Mean Squared Errors
mseStock = mean_squared_error(y_test, y_predict)
mseIris = mean_squared_error(y_testIris, y_predictIris)
mseSentiments = mean_squared_error(y_testSentiments, y_predictSentiments)

#Root Mean Squared Errors
rmseStock = root_mean_squared_error(y_test, y_predict)
rmseIris = root_mean_squared_error(y_testIris, y_predictIris)
rmseSentiments = root_mean_squared_error(y_testSentiments, y_predictSentiments)

#Mean Absolute Errors
maeStock = mean_absolute_error(y_test, y_predict)
maeIris = mean_absolute_error(y_testIris, y_predictIris)
maeSentiments = mean_absolute_error(y_testSentiments, y_predictSentiments)

#outputting the results ahead
print("Intercepts of Stock: ", model.intercept_)
print("Coefficients of Stock: ", model.coef_)

print("Intercepts of Iris: ", modelIris.intercept_)
print("Coefficients of Iris: ", modelIris.coef_)

print("Intercepts of Sentiments: ", modelSentiments.intercept_)
print("Coefficients of Sentiments: ", modelSentiments.coef_)

print(f"R-squared of Iris (through metrics): {r2MetricIris:.4f}")
print(f"R-squared of Iris (through model): {r2ModelScoreIris:.4f}")

print(f"R-squared of Stock (through metrics): {r2MetricStock:.4f}")
print(f"R-squared of Stock (through model): {r2ModelScoreStock:.4f}")

print(f"R-squared of Sentiments (through metrics): {r2MetricSentiments:.4f}")
print(f"R-squared of Sentiments (through model): {r2ModelScoreSentiments:.4f}")

print(f"Mean Squared Error of Stock: {mseStock:.2f}")
print(f"Mean Squared Error of Iris: {mseIris:.2f}")
print(f"Mean Squared Error of Sentiments: {mseSentiments:.2f}")

print(f"Mean Absolute Error of Stock: {maeStock:.2f}")
print(f"Mean Absolute Error of Iris: {maeIris:.2f}")
print(f"Mean Absolute Error of Sentiments: {maeSentiments:.2f}")

print(f"Root Mean Squared Error of Stock: {rmseStock:.2f}")
print(f"Root Mean Squared Error of Iris: {rmseIris:.2f}")
print(f"Root Mean Squared Error of Sentiments: {rmseSentiments:.2f}")

#creating the visual of the MLR
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_predict, color='blue', alpha=0.5, label='Points')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color= 'red', linestyle= '--', label='Prediction')

plt.xlabel('Actual values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Multiple Linear Regression)')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_testIris, y_predictIris, color='blue', alpha=0.5, label='Points')
plt.plot([y_testIris.min(), y_testIris.max()], [y_testIris.min(), y_testIris.max()], color= 'red', linestyle= '--', label='Prediction')

plt.xlabel('Actual values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Multiple Linear Regression)')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_testSentiments, y_predictSentiments, color='blue', alpha=0.5, label='Points')
plt.plot([y_testSentiments.min(), y_testSentiments.max()], [y_testSentiments.min(), y_testSentiments.max()], color= 'red', linestyle= '--', label='Prediction')

plt.xlabel('Actual values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Multiple Linear Regression)')
plt.legend()
plt.show()