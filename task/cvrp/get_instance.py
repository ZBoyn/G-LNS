import numpy as np

LOW_DEMAND = 1
HIGH_DEMAND = 9
DEPOT_COORDINATES = np.array([0.5, 0.5])

class GetData:
    def __init__(self, n_instance, n_cities, capacity):
        self.n_instance = n_instance
        self.n_cities = n_cities + 1 # Includes depot
        self.capacity = capacity

    def generate_instances(self):
        np.random.seed(2024)
        instance_data = []
        for _ in range(self.n_instance):
            coordinates = np.concatenate([DEPOT_COORDINATES[np.newaxis, :], np.random.rand(self.n_cities - 1, 2)])
            demands = np.concatenate([[0], np.random.randint(LOW_DEMAND, HIGH_DEMAND + 1, size=self.n_cities - 1)])
            distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)
            instance_data.append((coordinates, distances, demands, self.capacity))
        return instance_data


if __name__ == '__main__':
    gd = GetData(10, 50, 50)
    data = gd.generate_instances()