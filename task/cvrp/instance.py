import numpy as np

class CVRPInstance:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray, demands: np.ndarray, capacity: int, name: str = None, optimal_cost: float = None):
        self.coordinates = coordinates
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.capacity = capacity
        self.name = name
        self.optimal_cost = optimal_cost
        self.num_nodes = len(coordinates)
        self.depot_idx = 0

    def __repr__(self):
        return f"CVRPInstance(n={self.num_nodes}, cap={self.capacity}, name={self.name}"

