import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles

#uses the sklearn function to make a dataset 
# x,y = make_moons()
x, y = make_circles()

#sets the value of k and creates kmeans model
k = 2
km = KMeans(n_clusters=k).fit(x)

#returns centroid x, y values in a 2D array
centroids = km.cluster_centers_

#returns the cluster labels of each data point in the x_std data set
labels = km.labels_

#sets the size of the scatterplot
plt.figure(figsize=(5,4))

#plots the data points for each cluster
for i in range(k):
    cluster = x[labels == i]
    plt.scatter(cluster[:,0], cluster[:,1])

#plots the centriods
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=100,
            c='r', label='centroid')

plt.show()