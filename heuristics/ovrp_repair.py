import random
import math
# Best Solution: 5.69

# Score: 314.5
def repair_v1(x, removed_customers, problem_data):
    new_x = [route[:] for route in x]
    demands = problem_data['demands']
    capacity = problem_data['capacity']
    dist_mat = problem_data['distance_matrix']
    depot = problem_data.get('depot_idx', 0)
    
    # Helper: compute route load
    def route_load(route):
        return sum(demands[c] for c in route)
    
    # Helper: insertion cost for a customer at a position in a route
    def insertion_cost(route, cust, pos):
        prev_node = route[pos-1] if pos > 0 else depot
        next_node = route[pos] if pos < len(route) else None
        if next_node is None:
            return dist_mat[prev_node][cust]
        else:
            added = dist_mat[prev_node][cust] + dist_mat[cust][next_node]
            removed = dist_mat[prev_node][next_node]
            return added - removed
    
    # For each customer, compute best insertion costs for all routes
    # and store top-3 costs for regret calculation
    while removed_customers:
        best_costs = {}
        second_best_costs = {}
        third_best_costs = {}
        best_positions = {}
        best_routes = {}
        
        for cust in removed_customers:
            cust_demand = demands[cust]
            costs = []
            
            # Evaluate existing routes
            for r_idx, route in enumerate(new_x):
                if route_load(route) + cust_demand > capacity:
                    continue
                best_cost_route = float('inf')
                best_pos_route = -1
                for pos in range(len(route) + 1):
                    inc = insertion_cost(route, cust, pos)
                    if inc < best_cost_route:
                        best_cost_route = inc
                        best_pos_route = pos
                if best_cost_route < float('inf'):
                    costs.append((best_cost_route, r_idx, best_pos_route))
            
            # Evaluate new route
            new_route_cost = dist_mat[depot][cust]
            costs.append((new_route_cost, len(new_x), 0))
            
            # Sort by cost
            costs.sort(key=lambda t: t[0])
            
            if len(costs) >= 1:
                best_costs[cust] = costs[0][0]
                best_routes[cust] = costs[0][1]
                best_positions[cust] = costs[0][2]
            if len(costs) >= 2:
                second_best_costs[cust] = costs[1][0]
            else:
                second_best_costs[cust] = float('inf')
            if len(costs) >= 3:
                third_best_costs[cust] = costs[2][0]
            else:
                third_best_costs[cust] = float('inf')
        
        # Compute regret-3 values: sum of differences between best and k-th best
        regrets = {}
        for cust in removed_customers:
            regret = (second_best_costs[cust] - best_costs[cust]) + \
                     (third_best_costs[cust] - best_costs[cust])
            # Add a penalty based on the best route's current length to avoid overloading long routes
            best_route_idx = best_routes[cust]
            if best_route_idx < len(new_x):
                route_len = len(new_x[best_route_idx])
                penalty = 0.01 * best_costs[cust] * route_len
            else:
                penalty = 0
            regrets[cust] = regret - penalty
        
        # Select customer with maximum regret
        selected = max(removed_customers, key=lambda c: regrets[c])
        removed_customers.remove(selected)
        
        # Insert selected customer
        r_idx = best_routes[selected]
        pos = best_positions[selected]
        if r_idx == len(new_x):
            new_x.append([selected])
        else:
            new_x[r_idx].insert(pos, selected)
    
    return new_x

# Score: 357.4
def repair_v2(x, removed_customers, problem_data):
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
                next_node = route[pos] if pos < len(route) else None
                
                if next_node is None:
                    increase = dist_mat[prev_node][customer]
                else:
                    added = dist_mat[prev_node][customer] + dist_mat[customer][next_node]
                    removed = dist_mat[prev_node][next_node]
                    increase = added - removed
                
                if increase < best_cost_increase:
                    best_cost_increase = increase
                    best_route_idx = r_idx
                    best_insert_pos = pos
        
        new_route_cost = dist_mat[depot][customer]
        if new_route_cost < best_cost_increase:
            best_cost_increase = new_route_cost
            best_route_idx = len(new_x)
            best_insert_pos = 0
            
        if best_route_idx == len(new_x):
            new_x.append([customer])
        else:
            new_x[best_route_idx].insert(best_insert_pos, customer)
            
    return new_x

# Score: 297.7
def repair_v3(x, removed_customers, problem_data):
    new_x = [route[:] for route in x]
    demands = problem_data['demands']
    capacity = problem_data['capacity']
    dist_mat = problem_data['distance_matrix']
    depot = problem_data.get('depot_idx', 0)
    
    def compute_best_insertion(customer, routes):
        cust_demand = demands[customer]
        best_increase = float('inf')
        best_route_idx = -1
        best_pos = -1
        
        for r_idx, route in enumerate(routes):
            route_load = sum(demands[c] for c in route)
            if route_load + cust_demand > capacity:
                continue
            for pos in range(len(route) + 1):
                prev_node = route[pos-1] if pos > 0 else depot
                next_node = route[pos] if pos < len(route) else None
                if next_node is None:
                    increase = dist_mat[prev_node][customer]
                else:
                    added = dist_mat[prev_node][customer] + dist_mat[customer][next_node]
                    removed = dist_mat[prev_node][next_node]
                    increase = added - removed
                if increase < best_increase:
                    best_increase = increase
                    best_route_idx = r_idx
                    best_pos = pos
        new_route_cost = dist_mat[depot][customer]
        if new_route_cost < best_increase:
            best_increase = new_route_cost
            best_route_idx = len(routes)
            best_pos = 0
        return best_increase, best_route_idx, best_pos
    
    unplaced = list(removed_customers)
    while unplaced:
        candidate_data = []
        for customer in unplaced:
            cust_demand = demands[customer]
            feasible_insertions = []
            for r_idx, route in enumerate(new_x):
                route_load = sum(demands[c] for c in route)
                if route_load + cust_demand > capacity:
                    continue
                for pos in range(len(route) + 1):
                    prev_node = route[pos-1] if pos > 0 else depot
                    next_node = route[pos] if pos < len(route) else None
                    if next_node is None:
                        increase = dist_mat[prev_node][customer]
                    else:
                        added = dist_mat[prev_node][customer] + dist_mat[customer][next_node]
                        removed = dist_mat[prev_node][next_node]
                        increase = added - removed
                    feasible_insertions.append((increase, r_idx, pos))
            new_route_cost = dist_mat[depot][customer]
            feasible_insertions.append((new_route_cost, len(new_x), 0))
            
            if not feasible_insertions:
                candidate_data.append((customer, 0, -1, -1, 0))
                continue
                
            feasible_insertions.sort(key=lambda x: x[0])
            best_increase, best_route_idx, best_pos = feasible_insertions[0]
            regret = 0
            if len(feasible_insertions) >= 2:
                regret = feasible_insertions[1][0] - feasible_insertions[0][0]
            candidate_data.append((customer, regret, best_route_idx, best_pos, len(feasible_insertions)))
        
        candidate_data.sort(key=lambda item: (-item[4], -item[1]))
        chosen_customer, _, best_route_idx, best_pos, _ = candidate_data[0]
        
        if best_route_idx == len(new_x):
            new_x.append([chosen_customer])
        else:
            new_x[best_route_idx].insert(best_pos, chosen_customer)
        unplaced.remove(chosen_customer)
    
    return new_x

# Score: 148.8
def repair_v4(x, removed_customers, problem_data):
    def compute_insertion_cost(customer, route, pos, dist_mat, depot):
        if pos == 0:
            prev = depot
        else:
            prev = route[pos-1]
        if pos == len(route):
            next_node = None
        else:
            next_node = route[pos]
        
        if next_node is None:
            return dist_mat[prev][customer]
        else:
            added = dist_mat[prev][customer] + dist_mat[customer][next_node]
            removed = dist_mat[prev][next_node]
            return added - removed
    
    new_x = [route[:] for route in x]
    demands = problem_data['demands']
    capacity = problem_data['capacity']
    dist_mat = problem_data['distance_matrix']
    depot = problem_data.get('depot_idx', 0)
    
    unplaced = list(removed_customers)
    
    while unplaced:
        candidate_scores = []
        
        for customer in unplaced:
            cust_demand = demands[customer]
            feasible_options = []
            
            for r_idx, route in enumerate(new_x):
                route_load = sum(demands[c] for c in route)
                if route_load + cust_demand > capacity:
                    continue
                
                for pos in range(len(route) + 1):
                    cost = compute_insertion_cost(customer, route, pos, dist_mat, depot)
                    feasible_options.append((cost, r_idx, pos))
            
            new_route_cost = dist_mat[depot][customer]
            feasible_options.append((new_route_cost, len(new_x), 0))
            
            if not feasible_options:
                candidate_scores.append((customer, float('-inf'), 0, -1, -1))
                continue
            
            feasible_options.sort(key=lambda x: x[0])
            best_cost, best_route, best_pos = feasible_options[0]
            
            regret = 0
            if len(feasible_options) >= 2:
                regret = feasible_options[1][0] - feasible_options[0][0]
            
            route_utilization = 0
            if best_route < len(new_x):
                route_load = sum(demands[c] for c in new_x[best_route])
                route_utilization = route_load / capacity
            
            score = (len(feasible_options) * 10) + regret - (route_utilization * 5)
            candidate_scores.append((customer, score, best_cost, best_route, best_pos))
        
        candidate_scores.sort(key=lambda x: (-x[1], x[2]))
        chosen_customer, _, _, best_route, best_pos = candidate_scores[0]
        
        if best_route == len(new_x):
            new_x.append([chosen_customer])
        else:
            new_x[best_route].insert(best_pos, chosen_customer)
        
        unplaced.remove(chosen_customer)
    
    return new_x

# Score: 392.2
def repair_v5(x, removed_customers, problem_data):
    new_x = [route[:] for route in x]
    demands = problem_data['demands']
    capacity = problem_data['capacity']
    dist_mat = problem_data['distance_matrix']
    depot = problem_data.get('depot_idx', 0)
    
    def route_load(route):
        return sum(demands[c] for c in route)
    
    def insertion_cost(route, cust, pos):
        prev_node = route[pos-1] if pos > 0 else depot
        next_node = route[pos] if pos < len(route) else None
        if next_node is None:
            return dist_mat[prev_node][cust]
        else:
            added = dist_mat[prev_node][cust] + dist_mat[cust][next_node]
            removed = dist_mat[prev_node][next_node]
            return added - removed
    
    # Shuffle to avoid order bias
    random.shuffle(removed_customers)
    
    while removed_customers:
        candidate_data = []
        
        for cust in removed_customers:
            cust_demand = demands[cust]
            insertion_options = []
            
            # Evaluate all existing routes exhaustively (Parent 1 approach)
            for r_idx, route in enumerate(new_x):
                if route_load(route) + cust_demand > capacity:
                    continue
                for pos in range(len(route) + 1):
                    cost = insertion_cost(route, cust, pos)
                    insertion_options.append((cost, r_idx, pos))
            
            # New route option
            new_route_cost = dist_mat[depot][cust]
            insertion_options.append((new_route_cost, len(new_x), 0))
            
            # Sort by cost
            insertion_options.sort(key=lambda t: t[0])
            
            # Extract top 3 for regret calculation (Parent 2 inspired)
            if len(insertion_options) >= 1:
                best_cost, best_route, best_pos = insertion_options[0]
                second_best = insertion_options[1][0] if len(insertion_options) >= 2 else float('inf')
                third_best = insertion_options[2][0] if len(insertion_options) >= 3 else float('inf')
                
                # Regret-3 calculation
                regret = (second_best - best_cost) + (third_best - best_cost)
                
                # Add penalty based on route length to avoid overloading long routes
                if best_route < len(new_x):
                    route_len = len(new_x[best_route])
                    penalty = 0.005 * best_cost * route_len
                else:
                    penalty = 0
                
                adjusted_regret = regret - penalty
                candidate_data.append((cust, adjusted_regret, best_cost, best_route, best_pos))
            else:
                # No feasible insertion (should not happen with new route option)
                candidate_data.append((cust, -float('inf'), float('inf'), len(new_x), 0))
        
        # Sort by regret descending
        candidate_data.sort(key=lambda t: t[1], reverse=True)
        
        # Select from top 3 regret customers with probability favoring higher regret
        top_k = min(3, len(candidate_data))
        if top_k > 0:
            weights = [math.exp(2.0 * candidate_data[i][1]) for i in range(top_k)]
            if sum(weights) > 0:
                selected_idx = random.choices(range(top_k), weights=weights, k=1)[0]
            else:
                selected_idx = 0
        else:
            selected_idx = 0
        
        selected_cust, _, _, best_route, best_pos = candidate_data[selected_idx]
        removed_customers.remove(selected_cust)
        
        # Perform insertion
        if best_route == len(new_x):
            new_x.append([selected_cust])
        else:
            new_x[best_route].insert(best_pos, selected_cust)
    
    return new_x
