import random
import copy
import numpy as np

def destroy_v1(x, destroy_cnt, problem_data):
    new_x = [route[:] for route in x]
    removed_customers = []
    
    all_customers = [c for route in new_x for c in route]
    
    if len(all_customers) <= destroy_cnt:
        return all_customers, [[]]
        
    targets = random.sample(all_customers, destroy_cnt)
    
    for customer in targets:
        removed_customers.append(customer)
        for route in new_x:
            if customer in route:
                route.remove(customer)
                break
                
    new_x = [r for r in new_x if len(r) > 0]
    
    return removed_customers, new_x

def destroy_v2(x, destroy_cnt, problem_data):
    dist_mat = problem_data.get('distance_matrix')
    if dist_mat is None:
        return destroy_v1(x, destroy_cnt, problem_data)
        
    new_x = [route[:] for route in x]
    removed_customers = []
    depot = problem_data.get('depot_idx', 0)
    
    savings = []
    
    for r_idx, route in enumerate(new_x):
        for i, node in enumerate(route):
            prev_node = route[i-1] if i > 0 else depot
            next_node = route[i+1] if i < len(route)-1 else depot
            
            cost_with = dist_mat[prev_node][node] + dist_mat[node][next_node]
            cost_without = dist_mat[prev_node][next_node]
            
            saving = cost_with - cost_without
            savings.append((saving, node, r_idx))
            
    savings.sort(key=lambda x: x[0], reverse=True)
    
    targets_info = savings[:destroy_cnt]
    targets = [t[1] for t in targets_info]
    
    for customer in targets:
        removed_customers.append(customer)
        for route in new_x:
            if customer in route:
                route.remove(customer)
                break
                
    new_x = [r for r in new_x if len(r) > 0]
    return removed_customers, new_x

def destroy_v3(x, destroy_cnt, problem_data):
    dist_mat = problem_data.get('distance_matrix')
    if dist_mat is None:
        return destroy_v1(x, destroy_cnt, problem_data)
        
    new_x = [route[:] for route in x]
    
    all_customers = [c for route in new_x for c in route]
    
    if len(all_customers) <= destroy_cnt:
        return all_customers, [[]]
        
    center_customer = random.choice(all_customers)
    
    candidates = []
    for c in all_customers:
        dist = dist_mat[center_customer][c]
        candidates.append((c, dist))
        
    candidates.sort(key=lambda item: item[1])
    
    targets = [item[0] for item in candidates[:destroy_cnt]]
    
    removed_customers = []
    for customer in targets:
        removed_customers.append(customer)
        for route in new_x:
            if customer in route:
                route.remove(customer)
                break
                
    new_x = [r for r in new_x if len(r) > 0]
    return removed_customers, new_x

def insert_v1(x, removed_customers, problem_data):
    new_x = [route[:] for route in x]
    demands = problem_data['demands']
    capacity = problem_data['capacity']
    
    for customer in removed_customers:
        cust_demand = demands[customer]
        inserted = False
        
        candidate_routes = list(range(len(new_x)))
        random.shuffle(candidate_routes)
        
        for r_idx in candidate_routes:
            route = new_x[r_idx]
            route_load = sum(demands[c] for c in route)
            
            if route_load + cust_demand <= capacity:
                insert_pos = random.randint(0, len(route))
                route.insert(insert_pos, customer)
                inserted = True
                break
                
        if not inserted:
            new_x.append([customer])
            
    return new_x

def insert_v2(x, removed_customers, problem_data):
    new_x = [route[:] for route in x]
    demands = problem_data['demands']
    capacity = problem_data['capacity']
    dist_mat = problem_data['distance_matrix']
    depot = problem_data.get('depot_idx', 0)

    for customer in removed_customers:
        cust_demand = demands[customer]
        best_cost_increase = float('inf')
        best_route_idx = -1
        best_insert_pos = -1
        
        for r_idx, route in enumerate(new_x):
            route_load = sum(demands[c] for c in route)
            if route_load + cust_demand > capacity:
                continue
            
            for pos in range(len(route) + 1):
                prev_node = route[pos-1] if pos > 0 else depot
                next_node = route[pos] if pos < len(route) else depot
                
                added = dist_mat[prev_node][customer] + dist_mat[customer][next_node]
                removed = dist_mat[prev_node][next_node]
                
                increase = added - removed
                
                if increase < best_cost_increase:
                    best_cost_increase = increase
                    best_route_idx = r_idx
                    best_insert_pos = pos
        
        new_route_cost = dist_mat[depot][customer] + dist_mat[customer][depot]
        if new_route_cost < best_cost_increase:
            best_cost_increase = new_route_cost
            best_route_idx = len(new_x)
            best_insert_pos = 0
            
        if best_route_idx == len(new_x):
            new_x.append([customer])
        else:
            new_x[best_route_idx].insert(best_insert_pos, customer)
            
    return new_x

def insert_v3(x, removed_customers, problem_data):
    new_x = [route[:] for route in x]
    demands = problem_data['demands']
    capacity = problem_data['capacity']
    dist_mat = problem_data['distance_matrix']
    depot = problem_data.get('depot_idx', 0)
    
    sorted_removed = sorted(removed_customers, key=lambda c: demands[c], reverse=True)
    
    for customer in sorted_removed:
        cust_demand = demands[customer]
        best_cost_increase = float('inf')
        best_route_idx = -1
        best_insert_pos = -1
        
        for r_idx, route in enumerate(new_x):
            route_load = sum(demands[c] for c in route)
            if route_load + cust_demand > capacity:
                continue
            
            for pos in range(len(route) + 1):
                prev_node = route[pos-1] if pos > 0 else depot
                next_node = route[pos] if pos < len(route) else depot
                
                added = dist_mat[prev_node][customer] + dist_mat[customer][next_node]
                removed = dist_mat[prev_node][next_node]
                
                increase = added - removed
                
                if increase < best_cost_increase:
                    best_cost_increase = increase
                    best_route_idx = r_idx
                    best_insert_pos = pos
        
        new_route_cost = dist_mat[depot][customer] + dist_mat[customer][depot]
        if new_route_cost < best_cost_increase:
            best_cost_increase = new_route_cost
            best_route_idx = len(new_x)
            best_insert_pos = 0
            
        if best_route_idx == len(new_x):
            new_x.append([customer])
        else:
            new_x[best_route_idx].insert(best_insert_pos, customer)
            
    return new_x