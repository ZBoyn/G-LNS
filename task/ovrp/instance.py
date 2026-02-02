import numpy as np

class OVRPInstance:
    def __init__(self, coordinates: np.ndarray, distance_matrix: np.ndarray, demands: np.ndarray, capacity: int, name: str = None, optimal_cost: float = None):
        self.coordinates = coordinates
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.capacity = capacity
        self.name = name
        self.num_nodes = len(coordinates)
        self.depot_idx = 0
        self.optimal_cost = optimal_cost

    def __repr__(self):
        return f"OVRPInstance(n={self.num_nodes}, cap={self.capacity}, name={self.name})"
