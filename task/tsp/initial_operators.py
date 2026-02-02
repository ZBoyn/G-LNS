import random
import copy
import numpy as np

# ==================================================
#                   Destroy Operators
# ==================================================
def destroy_v1(x, destroy_cnt, problem_data):
    new_x = copy.deepcopy(x)
    removed_cities = []
    
    if len(x) <= destroy_cnt:
        return list(range(len(x))), []
        
    removed_index = random.sample(range(0, len(x)), destroy_cnt)
    removed_index.sort(reverse=True)
    
    for i in removed_index:
        removed_cities.append(new_x[i])
        new_x.pop(i)
        
    return removed_cities, new_x

def destroy_v2(x, destroy_cnt, problem_data):
    if problem_data is None:
        dist_mat = None
    elif isinstance(problem_data, dict):
        dist_mat = problem_data.get('distance_matrix')
    else:
        dist_mat = problem_data

    if dist_mat is None:
        return destroy_v1(x, destroy_cnt, problem_data)

    new_x = copy.deepcopy(x)
    removed_cities = []

    dis = []
    for i in range(len(new_x) - 1):
        dis.append(dist_mat[new_x[i]][new_x[i + 1]])
    dis.append(dist_mat[new_x[-1]][new_x[0]])
    
    sorted_edge_indices = np.argsort(np.array(dis))

    targets = []
    for i in range(len(sorted_edge_indices)):
        node_idx = sorted_edge_indices[-(i+1)]
        if node_idx < len(new_x):
            targets.append(node_idx)
        if len(targets) >= destroy_cnt:
            break
            
    targets.sort(reverse=True)
    for i in targets:
        removed_cities.append(new_x[i])
        new_x.pop(i)

    return removed_cities, new_x

def destroy_v3(x, destroy_cnt, problem_data=None):
    new_x = copy.deepcopy(x)
    removed_cities = []

    if len(x) <= destroy_cnt:
         return list(range(len(x))), []

    start_index = random.randint(0, len(x) - destroy_cnt)
    
    for i in range(start_index + destroy_cnt - 1, start_index - 1, -1):
        removed_cities.append(new_x[i])
        new_x.pop(i)
        
    return removed_cities, new_x


# ==================================================
#                   Insert Operators
# ==================================================
def insert_v1(x, removed_cities, problem_data=None):
    new_x = copy.deepcopy(x)
    for city in removed_cities:
        insert_index = random.randint(0, len(new_x))
        new_x.insert(insert_index, city)
    return new_x

def insert_v2(x, removed_cities, problem_data):
    if problem_data is None:
        dist_mat = None
    elif isinstance(problem_data, dict):
        dist_mat = problem_data.get('distance_matrix')
    else:
        dist_mat = problem_data
        
    if dist_mat is None:
        return insert_v1(x, removed_cities, problem_data)
        
    new_x = copy.deepcopy(x)
    
    def _calc_dist(path):
        d = 0
        for i in range(len(path) - 1):
            d += dist_mat[path[i]][path[i + 1]]
        if len(path) > 0:
            d += dist_mat[path[-1]][path[0]]
        return d

    for city in removed_cities:
        best_cost = float('inf')
        best_pos = -1
        
        for j in range(len(new_x) + 1):
            temp_x = new_x[:j] + [city] + new_x[j:]
            cost = _calc_dist(temp_x)
            if cost < best_cost:
                best_cost = cost
                best_pos = j
        
        new_x.insert(best_pos, city)
        
    return new_x
