import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# gets the data and sets x and y values
data = pd.read_csv("part1-linear-regression/chirping_data.csv")
x = data["Temp"].values
y = data["Chirps"].values

# use reshape to turn the x values into a 2D array
x = x.reshape(-1, 1)

# create the model
model = LinearRegression().fit(x, y)

# find the coefficient, bias, and r squared values
# each should be a float and rounded to two decimal places
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)

# value you are going to predict
x_predict = 77
# plug that value into your model
prediction = model.predict([[x_predict]])

# print out the linear equation and r squared value
print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")
print(f"Prediction when x is {x_predict}: {prediction}")

'''
The following code creates the graph to visualize the data
'''
# sets the size of the graph
plt.figure(figsize=(6,4))

# creates a scatter plot of originial data in purple
# and the predicted data in blue
plt.scatter(x,y, c="purple")
plt.scatter(x_predict, prediction, c="blue")

# labe the axes
plt.xlabel("Temperature °F")
plt.ylabel("Chirps per Minute")
plt.title("Cricket Chirps by Temperature")

# plot the line of best fit in red and label the line
plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

# show the plot and legend
plt.legend()
plt.show()