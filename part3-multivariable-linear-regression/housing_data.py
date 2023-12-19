from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
import time

print("Calidfornia housing dataset prediction.")
california = fetch_california_housing()
x, y = california.data, california.target
print(california)

x = scale(x)
y = scale(y)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.15)

print("")
print("Least Squares Linear Regression")
start = time.time()
model = LinearRegression().fit(xtrain, ytrain)
end = time.time()
print("Time to compute the model (in seconds): ", round((end-start), 3))
coef = model.coef_
print("Coefficients:")
print("MedInc:", round(coef[0], 2))
print("HouseAge:", round(coef[1], 2))
print("AveRooms", round(coef[2], 2))
print("Population:", round(coef[4], 2))
print("")

print("Intercept: ", model.intercept_)
print("Score: ", model.score(xtrain, ytrain))
print("")

print("*******************************")
print("Gradient Descent Linear Regression")
start = time.time()
sgdr = SGDRegressor().fit(xtrain, ytrain)
end = time.time()
print("Time to compute the model (in seconds): ", round((end-start), 3))
coef = sgdr.coef_
print(coef)
print("Coefficients:")
print("MedInc:", round(coef[0], 2))
print("HouseAge:", round(coef[1], 2))
print("AveRooms", round(coef[2], 2))
print("Population:", round(coef[4], 2))
print("")
print("Intercept: ", float(sgdr.intercept_))
print("Score: ", sgdr.score(xtrain, ytrain))