from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import numpy as np


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

import numpy as np


X, y = make_blobs(n_samples=500, centers=5, cluster_std = 1.00)
plt.figure(1)
plt.scatter(x=X[:,0], y=X[:,1], c =["green"])


model = KMeans(n_clusters=5, n_init='auto')
model.fit(X)
print(model.cluster_centers_)

plt.figure(2)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=100, c='black')

print(model.labels_)

plt.scatter(x=X[:,0], y=X[:,1], c= model.labels_, cmap='cool' )
plt.scatter(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1], c=['black'])
plt.show()