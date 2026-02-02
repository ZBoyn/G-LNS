import numpy as np

class TSPInstance:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray, name: str = None, optimal_cost: float = None):
        self.coordinates = coordinates
        self.distance_matrix = distance_matrix
        self.name = name
        self.num_cities = len(coordinates)
        self.optimal_cost = optimal_cost

    def __repr__(self):
        return f"TSPInstance(n={self.num_cities}, name={self.name})"
