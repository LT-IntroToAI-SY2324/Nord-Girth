import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#imports and formats the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles","age"]].values
y = data["Price"].values

#split the data into training and testing data

#create linear regression model

#Find and print the coefficients, intercept, and r squared values. 
#Each should be rounded to two decimal places. 


#Loop through the data and print out the predicted prices and the 
#actual prices
print("***************")
print("Testing Results")