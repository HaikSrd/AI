import numpy as np

class K_means:
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.num_classes = k

        self.centroids = np.empty((k, dataset.shape[1]))
        self.centroids[0] = self.dataset[np.random.choice(len(self.dataset))]
        self.distances = np.array([self.distance(self.centroids[0], i) for i in self.dataset])

    @staticmethod
    def distance(point1, point2):
        return np.linalg.norm(point1 - point2) ** 2

    def init_centroids(self):
        for k in range(1,self.num_classes):
            self.centroids[k] = self.dataset[np.random.choice(len(self.dataset), p = self.distances/np.sum(self.distances))]
            for i in range(len(self.dataset)):
                new_dist = self.distance(self.centroids[k], self.dataset[i])
                if new_dist < self.distances[i]:
                    self.distances[i] = new_dist
        return self.centroids

    def min_distance(self, point):
        dist = np.empty(self.num_classes)
        for i in range(self.num_classes):
            dist[i] = self.distance(self.centroids[i], point)
        return np.argmin(dist)


    def train(self, epochs):
        self.init_centroids()

        for epoch in range(epochs):
            labels = np.empty(len(self.dataset))
            for i in range(len(self.dataset)):
                labels[i] = self.min_distance(self.dataset[i])


            for k in range(self.num_classes):
                num_points = 0
                centre = np.zeros(self.dataset.shape[1])
                for i in range(len(self.dataset)):
                    if labels[i] == k:
                        centre += self.dataset[i]
                        num_points += 1
                self.centroids[k] = centre/num_points

        return self.centroids
