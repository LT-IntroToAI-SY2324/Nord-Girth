import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#imports the data
data = pd.read_csv("part5-unsupervised-learning/customer_data.csv")
x = data[["Annual Income", "Spending Score"]]

#standardize the data


#the value of k has been defined for you
k = 5

#apply the kmeans algorithm


#get the centroid and label values


#sets the size of the graph
plt.figure(figsize=(5,4))

#use a for loop to plot the data points in each cluster


#plot the centroids

            
#shows the graph
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()