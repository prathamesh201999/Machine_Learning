#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing dataset
df = pd.read_csv('Mall.csv')
x = df.iloc[:,[3,4]].values

#choosing value for k
import scipy.cluster.hierarchy as sch
dend = sch.dendrogram(sch.linkage(x, method='ward'))
plt.title('The Elbow Method')
plt.xlabel('K value')
plt.ylabel('WCSS')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity = 'euclidean',linkage = 'ward')
y_hc = hc.fit_predict(x)


#visualizing 
plt.scatter(x[y_hc == 0,0],x[y_hc == 0,1],s = 100,c = 'red', label = 'Cluster1')
plt.scatter(x[y_hc == 1,0],x[y_hc == 1,1],s = 100,c = 'blue', label = 'Cluster2')
plt.scatter(x[y_hc == 2,0],x[y_hc == 2,1],s = 100,c = 'orange', label = 'Cluster3')
plt.scatter(x[y_hc == 3,0],x[y_hc == 3,1],s = 100,c = 'green', label = 'Cluster4')
plt.scatter(x[y_hc == 4,0],x[y_hc == 4,1],s = 100,c = 'pink', label = 'Cluster5')
plt.scatter(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],s = 300,c = 'yellow', label = 'Centroid')
plt.title('Heirarchy Clustering')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.legend()
plt.show()
