import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import silhouette_score


iris = datasets.load_iris()
X = iris.data
y = iris.target


k = 3  
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)


labels = kmeans.labels_


print("Silhouette score for k-means is: ", (silhouette_score(X, labels, metric='euclidean')))


k = 3 
kmedoids = KMedoids(n_clusters=k, random_state=0).fit(X)


labels = kmedoids.labels_


print("Silhouette score for kmedoids is: ", (silhouette_score(X, labels, metric='euclidean')))



plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('KMeans Clustering of Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()


plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('KMedoids Clustering of Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()
