from sklearn.datasets import make_blobs
from Kmeans import K_means

X, y = make_blobs(n_samples=300, centers=3, n_features=3, random_state=42)
kmeans = K_means(X, 3)
centres = kmeans.train(1000)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='viridis', alpha = 0.3)
ax.scatter(centres[:,0], centres[:,1], centres[:,2], color='red', s=100)
plt.show()


