import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Run the program and consider the following questions:
1. Look at the data points on the graph. Do age and blood pressure appear to have a linear relationship?
2. What does the value of r tell you about the relationship between age and blood pressure?
'''

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
x = data["Age"]
y = data["Blood Pressure"]

#sets the size of the graph
plt.figure(figsize=(5,4))

#labels axes and creates scatterplot
plt.xlabel("Age")
plt.ylabel("Systolic Blood Pressure")
plt.title("Systolic Blood Pressure by Age")
plt.scatter(x, y)

print("Pearson's Correlation: r = :", x.corr(y))

plt.show()