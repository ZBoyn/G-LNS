destroy_template_program = '''
import numpy as np
import random
import copy
def destroy_operator(x: list, destroy_cnt: int, dist_mat: np.ndarray = None) -> tuple:
    """
    Randomly remove a specific number of cities from the current solution.
    
    Args:
    x: The current solution (sequence of city IDs).
    destroy_cnt: The number of cities to remove.
    problem_data: Dictionary containing problem data, including 'distance_matrix'.
    
    Return:
    A tuple containing (removed_cities, new_x), where new_x is the partial solution.
    """
    # dist_mat = problem_data.get('distance_matrix')
    
    # Randomly remove N cities
    new_x = copy.deepcopy(x)
    removed_cities = []
    
    # Randomly select N cities to remove
    if len(x) <= destroy_cnt:
        return list(range(len(x))), [] # Edge case: remove everything
        
    removed_index = random.sample(range(0, len(x)), destroy_cnt)
    
    # To avoid index shifting issues during removal, sort indices in descending order
    removed_index.sort(reverse=True)
    
    for i in removed_index:
        removed_cities.append(new_x[i])
        new_x.pop(i) # pop by index
        
    return removed_cities, new_x
'''

destroy_task_description = """
The task is to design a novel **Destroy Operator** for a Large Neighborhood Search LNS framework for the Traveling Salesman Problem (TSP).
Given a complete solution sequence (a tour of cities for TSP) and a target number of elements to remove (`destroy_cnt`), the function must determine which elements to remove.
The objective is to develop a removal strategy that effectively perturbs the current solution. This allows the subsequent Repair operator to reconstruct the solution in a way that helps escape local optima and minimizes the total cost.

Data available in `problem_data`:
- `distance_matrix`: 2D numpy array of distances.
- `coordinates`: 2D numpy array of coordinates (N, 2).

Note on Solution Structure `x`:
`x` is a list of city IDs (integers) representing the tour, e.g., `[0, 5, 2, 3, 1, 4]`.
"""

insert_template_program = """
import numpy as np
import random
import copy
def insert_v1(x, removed_cities, problem_data):
    “”“
    Randomly insert removed cities back into the partial solution.

    Args:
    x: The partial solution (sequence of city IDs) after the destroy operation.
    removed_cities: List of cities that need to be re-inserted.
    problem_data: Dictionary containing problem data, including 'distance_matrix'.

    Return:
    The complete solution with all cities inserted.
    ”“”
    # dist_mat = problem_data.get('distance_matrix')
    
    new_x = copy.deepcopy(x)
    # Randomly insert each removed city into the solution
    for city in removed_cities:
        insert_index = random.randint(0, len(new_x))
        new_x.insert(insert_index, city)
    return new_x
"""

insert_task_description = """
The task is to design a novel **Repair Operator** (Insertion Operator) for a Large Neighborhood Search (LNS) framework for the Traveling Salesman Problem (TSP).
Given a partial solution `x` (where some elements have been removed) and a list of `removed_cities`, the function must determine the best positions to re-insert these elements to restore a complete solution.
The objective is to reconstruct the solution in a way that minimizes the total cost, total travel distance.

Data available in `problem_data`:
- `distance_matrix`: 2D numpy array of distances.
- `coordinates`: 2D numpy array of coordinates (N, 2).

Note on Solution Structure `x`:
`x` is a list of city IDs (integers) representing the partial tour.
"""