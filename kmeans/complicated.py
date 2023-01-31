
# https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE



digits = load_digits()
print("Digits data shape: {}".format(digits.data.shape))

kmeans = KMeans(n_clusters=10, random_state=0, n_init='auto')
clusters = kmeans.fit_predict(digits.data)
print("Kmeans cluster centers: {}".format(kmeans.cluster_centers_.shape))

kmeans = KMeans(n_clusters=10, random_state=0, n_init='auto')
clusters = kmeans.fit_predict(digits.data)
print("Kmeans cluster centers: {}".format(kmeans.cluster_centers_.shape))

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask], keepdims=True)[0]

print("Accuracy score: {}".format(accuracy_score(digits.target, labels)))


mat = confusion_matrix(digits.target, labels)
plt.imshow(mat.T)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

# Project the data: this step will take several seconds
tsne = TSNE(n_components=2, init='random', random_state=0)
digits_proj = tsne.fit_transform(digits.data)

# Compute the clusters
kmeans = KMeans(n_clusters=10, random_state=0, n_init='auto')
clusters = kmeans.fit_predict(digits_proj)

# Permute the labels
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask], keepdims=True)[0]

# Compute the accuracy
print("Accuracy score: {}".format(accuracy_score(digits.target, labels)))



# We will start by loading the digits and then finding the KMeans clusters. Recall that the digits consist of 1,797 samples with 64 features, where each of the 64 features is the brightness of one pixel in an 8×8 image:
# What if we jsut said each of the cells was the hash value and went on from there?
# Otherwise we could leave it as is, where each value describes the distance between any two values