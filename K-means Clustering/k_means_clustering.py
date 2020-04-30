#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing dataset
df = pd.read_csv('Mall.csv')
x = df.iloc[:,[3,4]].values

#choosing value for k
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmean = KMeans(n_clusters= i, init='k-means++',max_iter= 300, n_init=10,random_state = 0)
    kmean.fit(x)
    wcss.append(kmean.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('K value')
plt.ylabel('WCSS')
plt.show()

kmean = KMeans(n_clusters=5,n_init=10,init = 'k-means++',max_iter=300,random_state = 0)
y_means = kmean.fit_predict(x)



#visualizing 
plt.scatter(x[y_means == 0,0],x[y_means == 0,1],s = 100,c = 'red', label = 'Cluster1')
plt.scatter(x[y_means == 1,0],x[y_means == 1,1],s = 100,c = 'blue', label = 'Cluster2')
plt.scatter(x[y_means == 2,0],x[y_means == 2,1],s = 100,c = 'orange', label = 'Cluster3')
plt.scatter(x[y_means == 3,0],x[y_means == 3,1],s = 100,c = 'green', label = 'Cluster4')
plt.scatter(x[y_means == 4,0],x[y_means == 4,1],s = 100,c = 'pink', label = 'Cluster5')
plt.scatter(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],s = 300,c = 'yellow', label = 'Centroid')
plt.title('K-Means Clustering')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.legend()
plt.show()
