import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = np.array([[15, 35], [17, 46], [12, 80], [20, 50], [25, 70], [10, 40]])

kmeans =  KMeans(n_clusters=2)
kmeans.fit(x)
label = kmeans.labels_
print("clusters", label)

plt.scatter(x[:,0], x[:,1], c = label, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            color='blue', marker='x', s=200, label='centroid')

plt.xlabel("Annual Income")
plt.ylabel("Spending score")
plt.legend()
plt.show()
