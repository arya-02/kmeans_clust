#importing required libraires
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#creating 2d array using numpy
data=np.array([[5,3],[10,15],[15,12],[24,10],[30,45],[85,70],[71,80],[60,78],[55,52],[80,91]])

#plotting the array before training
plt.scatter(data[:,0],data[:,1],label="True Positions")
plt.show()

#training  the model
kmeans=KMeans(n_clusters=2)

#printing cluster centers and cluster labels
kmeans.fit(data)
print(kmeans.cluster_centers_)
print(kmeans.labels_)

#plotting the data after traing with two centers signifying that our data has been divided into two clusters
plt.scatter(data[:,0],data[:,1],c=kmeans.labels_,cmap="rainbow")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="black")

#predicting
plt.show()
pred=kmeans.predict([[20,17]])
print(pred)