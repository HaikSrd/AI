import numpy as np

"""
The class KNearestNeighbors takes two inputs, features and labels, to find the best value of K, call the 
train function in the class with a range of different values of k such as [1,3,5,7,...]
"""

class KNearestNeighbors:
    def __init__(self, features, values):
        self.features = features
        self.values = values
        self.num_features = len(features)

    @staticmethod
    def distance(point1, point2):
        return np.linalg.norm(point1 - point2)

    # calculates the k closest data points to a point we tell it
    @staticmethod
    def closest_points(k, distances, point):
        new_list = []
        for i in distances:
            if i[0] == point:
                new_list.append([i[1],i[2]])
            elif i[1] == point:
                new_list.append([i[0], i[2]])
        return [x[0] for x in sorted(new_list, key = lambda x: x[1])[:k]]

    def train(self, values_of_k):
        # calculating the distances between every 2 data points
        distances = []
        for i in range(self.num_features):
            for j in range(i + 1, self.num_features):
                distances.append([i, j, self.distance(self.features[i], self.features[j])])
        distances = np.array(distances)
        accurecy = []
        for k in values_of_k:
            correct = 0

            for feature in range(self.num_features):
                close_points = self.closest_points(k, distances, feature);
                answers = []
                for i in close_points:
                    answers.append(self.values[int(i)])
                guess = max(set(answers), key=answers.count)
                if guess == self.values[feature]:
                    correct += 1
            acc = round(100 * correct / self.num_features, 2)
            print(f"For k = {k}, Accurecy = {acc}%")
            accurecy.append(acc)
        print(f"Best K = {values_of_k[accurecy.index(max(accurecy))]}")
        return values_of_k[accurecy.index(max(accurecy))]
