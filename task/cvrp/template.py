destroy_template_program = '''
import numpy as np
import random
import copy

def destroy_operator(x: list, destroy_cnt: int, problem_data: dict) -> tuple:
    """
    Randomly remove a specific number of customers from the current solution.
    
    Args:
    x: The current solution (list of routes, where each route is a list of customer IDs).
       Example: [[1, 2, 3], [4, 5]] means two routes: 0->1->2->3->0 and 0->4->5->0.
    destroy_cnt: The number of customers to remove.
    problem_data: Dict containing 'distance_matrix', 'demands', 'capacity', 'depot_idx', 'coordinates'.
    
    Return:
    A tuple containing (removed_customers, new_x), where new_x is the partial solution.
    """
    # Randomly remove N customers
    new_x = [route[:] for route in x] # Deep copy of routes
    removed_customers = []
    
    # Flatten current solution to easily select customers
    all_customers = [c for route in new_x for c in route]
    
    if len(all_customers) <= destroy_cnt:
        return all_customers, [[]] # Edge case: remove everything
        
    # Randomly select customers to remove
    targets = random.sample(all_customers, destroy_cnt)
    
    # Remove them from routes
    for customer in targets:
        removed_customers.append(customer)
        for route in new_x:
            if customer in route:
                route.remove(customer)
                break
    
    # Clean up empty routes
    new_x = [route for route in new_x if len(route) > 0]
    if not new_x:
        new_x = [[]]
        
    return removed_customers, new_x
'''

destroy_task_description = """
The task is to design a novel **Destroy Operator** for the Capacitated Vehicle Routing Problem (CVRP) in a GLNS(Generative Large Neighborhood Search) framework.
Given a solution `x` (a list of routes, where each route is a list of customer IDs) and a `destroy_cnt`, the function must remove `destroy_cnt` customers.
The objective is to perturb the solution effectively (e.g., random removal, worst-distance removal, cluster removal) to allow the Repair operator to find better solutions.
Data available in `problem_data`:
- `distance_matrix`: 2D numpy array of distances.
- `demands`: 1D numpy array of customer demands.
- `capacity`: Vehicle capacity (int).
- `depot_idx`: Index of the depot (0).
- `coordinates`: 2D numpy array of coordinates (N, 2), Node 0 is the Depot at [0.5, 0.5].

Note on Solution Structure `x`:
`x` is a list of lists, e.g., `[[1, 2], [3, 4]]`.
- This implies two routes: Depot -> 1 -> 2 -> Depot, and Depot -> 3 -> 4 -> Depot.
- The Depot node (0) is NOT explicitly included in the lists, but is implied at the start and end of each route.
"""

insert_template_program = '''
import numpy as np
import random
import copy

def repair_operator(x: list, removed_customers: list, problem_data: dict) -> list:
    """
    Randomly insert removed customers back into the partial solution, respecting capacity.
    
    Args:
    x: The partial solution (list of routes).
    removed_customers: List of customers to re-insert.
    problem_data: Dict containing 'distance_matrix', 'demands', 'capacity', 'depot_idx', 'coordinates'.
    
    Return:
    The complete solution with all customers inserted.
    """
    new_x = [route[:] for route in x]
    demands = problem_data['demands']
    capacity = problem_data['capacity']
    
    # Simple random insertion
    for customer in removed_customers:
        cust_demand = demands[customer]
        inserted = False
        
        # Try to insert into existing routes
        random.shuffle(new_x) # Shuffle routes to try randomly
        for route in new_x:
            route_load = sum(demands[c] for c in route)
            if route_load + cust_demand <= capacity:
                insert_pos = random.randint(0, len(route))
                route.insert(insert_pos, customer)
                inserted = True
                break
        
        # If not inserted, create a new route
        if not inserted:
            new_x.append([customer])
            
    return new_x
'''

insert_task_description = """
The task is to design a novel **Repair Operator** for CVRP in a GLNS(Generative Large Neighborhood Search) framework.
Given a partial solution `x` (list of routes) and `removed_customers`, insert them back.
Constraints:
1. **Capacity**: The sum of demands in any route must not exceed `capacity`.
2. **Validity**: All customers must be served.
Objectives: Minimize total travel distance.
Strategies: Greedy insertion, Regret-based insertion, Spatial insertion (using coordinates), etc.
"""

