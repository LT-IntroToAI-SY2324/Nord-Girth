'''
Source/Attribution: 
Imad Dabbura
https://gist.github.com/ImadDabbura/30bf36136221b1ef3ea99586a825c486#file-kmeans_plot_img_compression-py
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

# Read the image
img = mpimg.imread('part5-unsupervised-learning/pup.jpg')
print(img)
img_size = img.shape
print(img_size)

X = img.reshape(img_size[0] * img_size[1], img_size[2])

# Run the Kmeans algorithm
km = KMeans(n_clusters=24, n_init = 10)
km.fit(X)

# Use the centroids to compress the image
X_compressed = km.cluster_centers_[km.labels_]
X_compressed = np.clip(X_compressed.astype('uint8'), 0, 255)

# Reshape X_recovered to have the same dimension as the original image 128 * 128 * 3
X_compressed = X_compressed.reshape(img_size[0], img_size[1], img_size[2])

# Plot the original and the compressed image next to each other
fig, ax = plt.subplots(1, 2, figsize = (5, 4))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[1].imshow(X_compressed)
ax[1].set_title('Compressed Image')
# Turn off the axes
for ax in fig.axes:
    ax.axis('off')
plt.tight_layout()
plt.show()