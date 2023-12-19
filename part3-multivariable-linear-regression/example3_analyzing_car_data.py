import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x_1 = data["miles"]
x_2 = data["age"]
y = data["Price"]

fig, graph = plt.subplots(2)
graph[0].scatter(x_1, y)
graph[0].set_xlabel("Total Miles")
graph[0].set_ylabel("Price")

graph[1].scatter(x_2, y)
graph[1].set_ylabel("Price")
graph[1].set_xlabel("Car Age")

print(f"Correlation between Total Miles and Car Price: {round(x_1.corr(y), 3)}")
print(f"Correlation between Age and Car Price: {round(x_2.corr(y), 3)}")

plt.tight_layout()
plt.show()