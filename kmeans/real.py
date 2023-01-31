#import dependencies
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

#load dataset
X = pd.read_csv('customer.csv')
X.head()
print(np.shape(X))
#filter the columns
X = X.filter(["Annual Income (k$)", "Spending Score (1-100)" ], axis = 1)
print(np.shape(X))
#plot the raw data
plt.figure(1)
plt.scatter(data = X, x="Annual Income (k$)", y= "Spending Score (1-100)", c = ["green"])


#apply scikit-learn kmeans clustering
model = KMeans(n_clusters= 5)
model.fit(X)

#print the centroids
print(model.cluster_centers_)

#plot the clustered data with centroids overlaid
plt.figure(2)
plt.scatter(data = X, x="Annual Income (k$)", y= "Spending Score (1-100)", c= model.labels_, cmap= 'rainbow' )
plt.scatter(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1], c=['black'])

#create dataframe with clustered labels
segmented_data = pd.DataFrame()
segmented_data['Customer Number'] = X.index.values
segmented_data['Segment'] = model.labels_

#Save segment 3 as a dataframe and display head
segmented_data = segmented_data[segmented_data.Segment==3]
segmented_data.head()

plt.show()