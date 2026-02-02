# Best Solution: 7.84

# Score: 341.3
def repair_v1(x, removed_customers, problem_data):
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

# Score: 352.1
def repair_v2(x, removed_customers, problem_data):
    new_x = [route[:] for route in x]
    demands = problem_data['demands']
    capacity = problem_data['capacity']
    dist_mat = problem_data['distance_matrix']
    depot = problem_data.get('depot_idx', 0)
    coords = problem_data.get('coordinates', None)
    
    # Helper: route load
    def route_load(route):
        return sum(demands[node] for node in route)
    
    # Helper: route radial distance (distance from depot to farthest customer)
    def route_radial(route):
        if not route:
            return 0
        return max(dist_mat[depot][node] for node in route)
    
    # Helper: insertion cost with route shape consideration
    def insertion_cost(route, pos, customer):
        if not route:
            return dist_mat[depot][customer] + dist_mat[customer][depot]
        
        prev_node = route[pos-1] if pos > 0 else depot
        next_node = route[pos] if pos < len(route) else depot
        base_cost = dist_mat[prev_node][customer] + dist_mat[customer][next_node] - dist_mat[prev_node][next_node]
        
        # Add penalty if insertion disrupts route compactness
        if coords is not None and len(route) > 1:
            if pos > 0 and pos < len(route):
                # Check if customer lies between prev and next
                prev_coord = coords[prev_node]
                next_coord = coords[next_node]
                cust_coord = coords[customer]
                
                # Calculate deviation from direct line
                import math
                def point_line_distance(px, py, x1, y1, x2, y2):
                    num = abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1)
                    den = math.sqrt((y2-y1)**2 + (x2-x1)**2)
                    return num/den if den > 0 else 0
                
                deviation = point_line_distance(cust_coord[0], cust_coord[1],
                                               prev_coord[0], prev_coord[1],
                                               next_coord[0], next_coord[1])
                base_cost += deviation * 0.1  # Small penalty for deviation
        
        return base_cost
    
    # Helper: route attractiveness score
    def route_attractiveness(route, customer):
        if not route:
            return 1.0  # New route attractiveness
        
        load = route_load(route)
        spare_capacity = capacity - load - demands[customer]
        
        if spare_capacity < 0:
            return 0.0
        
        # Normalize spare capacity
        spare_ratio = spare_capacity / capacity
        
        # Calculate proximity to route
        if coords is not None:
            # Find nearest customer in route
            min_dist = min(dist_mat[customer][node] for node in route)
            max_dist = max(dist_mat[customer][node] for node in route)
            proximity = 1.0 - (min_dist / (max_dist + 1e-10))
        else:
            proximity = 0.5
        
        # Combine factors
        return 0.6 * spare_ratio + 0.4 * proximity
    
    # Step 1: Dynamic customer clustering
    if len(new_x) > 0 and len(removed_customers) > 1:
        # Group customers by proximity to existing routes
        customer_groups = []
        ungrouped = list(removed_customers)
        
        while ungrouped:
            seed = ungrouped[0]
            group = [seed]
            ungrouped.remove(seed)
            
            # Find customers close to this seed
            to_remove = []
            for cust in ungrouped:
                if dist_mat[seed][cust] < 0.2 * route_radial([seed]):  # Adaptive threshold
                    group.append(cust)
                    to_remove.append(cust)
            
            for cust in to_remove:
                ungrouped.remove(cust)
            
            customer_groups.append(group)
        
        # Sort groups by size (largest first) then by distance to nearest route
        def group_priority(group):
            avg_dist = float('inf')
            if new_x:
                total_dist = 0
                for cust in group:
                    min_cust_dist = min(
                        min(dist_mat[cust][node] for node in route if route)
                        for route in new_x if route
                    ) if any(route for route in new_x) else 0
                    total_dist += min_cust_dist
                avg_dist = total_dist / len(group)
            return (-len(group), avg_dist)  # Larger groups first
        
        customer_groups.sort(key=group_priority)
        insertion_order = []
        for group in customer_groups:
            # Sort within group by demand (largest first)
            group.sort(key=lambda c: demands[c], reverse=True)
            insertion_order.extend(group)
    else:
        insertion_order = list(removed_customers)
    
    # Step 2: Multi-criteria regret insertion with route balancing
    uninserted = insertion_order[:]
    
    while uninserted:
        # Calculate current route statistics
        route_loads = [route_load(route) for route in new_x]
        avg_load = sum(route_loads) / len(route_loads) if route_loads else 0
        
        best_customer = None
        best_score = -float('inf')
        best_insert_info = None
        
        for customer in uninserted:
            cust_demand = demands[customer]
            feasible_options = []
            
            # Evaluate existing routes
            for r_idx, route in enumerate(new_x):
                if route_load(route) + cust_demand > capacity:
                    continue
                
                # Find best position in this route
                best_pos_cost = float('inf')
                best_pos = -1
                
                for pos in range(len(route) + 1):
                    cost = insertion_cost(route, pos, customer)
                    if cost < best_pos_cost:
                        best_pos_cost = cost
                        best_pos = pos
                
                if best_pos != -1:
                    attractiveness = route_attractiveness(route, customer)
                    load_balance = 1.0 - abs((route_load(route) + cust_demand - avg_load) / capacity)
                    combined_score = attractiveness * load_balance
                    
                    # Adjust cost by route fullness (prefer fuller routes for efficiency)
                    fullness = (route_load(route) + cust_demand) / capacity
                    adjusted_cost = best_pos_cost * (1.0 - 0.3 * fullness)  # Discount for fuller routes
                    
                    feasible_options.append((adjusted_cost, combined_score, r_idx, best_pos))
            
            # New route option
            new_route_cost = insertion_cost([], 0, customer)
            feasible_options.append((new_route_cost, 1.0, len(new_x), 0))
            
            if not feasible_options:
                continue
            
            # Sort by cost
            feasible_options.sort(key=lambda t: t[0])
            
            # Calculate dynamic regret (consider top 3 options)
            k = min(3, len(feasible_options))
            regret_sum = 0
            for i in range(1, k):
                regret_sum += feasible_options[i][0] - feasible_options[0][0]
            
            if k > 1:
                avg_regret = regret_sum / (k - 1)
            else:
                avg_regret = 0
            
            # Combine regret with route attractiveness and customer priority
            customer_priority = 1.0
            if coords is not None and new_x:
                # Customers far from all routes get higher priority
                min_dist_to_any = min(
                    min(dist_mat[customer][node] for node in route if route)
                    for route in new_x if route
                ) if any(route for route in new_x) else 0
                customer_priority = 1.0 + min_dist_to_any / 100.0
            
            # Final score: regret + attractiveness + priority
            base_score = avg_regret * 0.5 + feasible_options[0][1] * 0.3 + customer_priority * 0.2
            
            # Penalize if customer has high demand and few feasible routes
            feasible_count = len([opt for opt in feasible_options if opt[0] < float('inf')])
            demand_penalty = 1.0 - min(1.0, feasible_count / 5.0) if cust_demand > capacity * 0.3 else 0.0
            final_score = base_score * (1.0 - demand_penalty * 0.2)
            
            if final_score > best_score:
                best_score = final_score
                best_customer = customer
                best_insert_info = feasible_options[0]
        
        # Insert best customer
        if best_customer is not None:
            _, _, best_route_idx, best_pos = best_insert_info
            if best_route_idx == len(new_x):
                new_x.append([best_customer])
            else:
                new_x[best_route_idx].insert(best_pos, best_customer)
            uninserted.remove(best_customer)
        else:
            # Emergency insertion: create new routes for remaining customers
            for customer in uninserted[:]:
                new_x.append([customer])
                uninserted.remove(customer)
    
    # Step 3: Local optimization of inserted customers
    for route in new_x:
        if len(route) <= 2:
            continue
        
        # Try to improve positions of recently inserted customers
        improved = True
        while improved:
            improved = False
            for i in range(len(route)):
                customer = route[i]
                if customer not in removed_customers:
                    continue
                
                # Try all other positions in this route
                current_cost = 0
                if i > 0:
                    current_cost += dist_mat[route[i-1]][customer]
                if i < len(route) - 1:
                    current_cost += dist_mat[customer][route[i+1]]
                
                best_pos = i
                best_cost = current_cost
                
                for j in range(len(route)):
                    if j == i:
                        continue
                    
                    # Calculate cost if customer were at position j
                    temp_route = route[:i] + route[i+1:]
                    if j < len(temp_route):
                        temp_route.insert(j, customer)
                    else:
                        temp_route.append(customer)
                    
                    new_cost = 0
                    for k in range(len(temp_route)):
                        if k == 0:
                            new_cost += dist_mat[depot][temp_route[k]]
                        else:
                            new_cost += dist_mat[temp_route[k-1]][temp_route[k]]
                    if temp_route:
                        new_cost += dist_mat[temp_route[-1]][depot]
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_pos = j
                
                if best_pos != i:
                    # Move customer to better position
                    route.pop(i)
                    if best_pos < len(route):
                        route.insert(best_pos, customer)
                    else:
                        route.append(customer)
                    improved = True
                    break
    
    return new_x

# Score: 353.0
def repair_v3(x, removed_customers, problem_data):
    new_x = [route[:] for route in x]
    demands = problem_data['demands']
    capacity = problem_data['capacity']
    dist_mat = problem_data['distance_matrix']
    depot = problem_data.get('depot_idx', 0)
    coords = problem_data.get('coordinates', None)
    
    # Helper: route load
    def route_load(route):
        return sum(demands[node] for node in route)
    
    # Helper: route radial distance
    def route_radial(route):
        if not route:
            return 0
        return max(dist_mat[depot][node] for node in route)
    
    # Helper: insertion cost
    def insertion_cost(route, pos, customer):
        if not route:
            return dist_mat[depot][customer] + dist_mat[customer][depot]
        prev_node = route[pos-1] if pos > 0 else depot
        next_node = route[pos] if pos < len(route) else depot
        return dist_mat[prev_node][customer] + dist_mat[customer][next_node] - dist_mat[prev_node][next_node]
    
    # Helper: route attractiveness
    def route_attractiveness(route, customer):
        if not route:
            return 1.0
        load = route_load(route)
        spare = capacity - load - demands[customer]
        if spare < 0:
            return 0.0
        spare_ratio = spare / capacity
        if coords is not None:
            min_dist = min(dist_mat[customer][node] for node in route)
            max_dist = max(dist_mat[customer][node] for node in route)
            proximity = 1.0 - (min_dist / (max_dist + 1e-10))
        else:
            proximity = 0.5
        return 0.6 * spare_ratio + 0.4 * proximity
    
    # Step 1: Dynamic customer clustering (simplified)
    if len(new_x) > 0 and len(removed_customers) > 1:
        ungrouped = list(removed_customers)
        customer_groups = []
        while ungrouped:
            seed = ungrouped[0]
            group = [seed]
            ungrouped.remove(seed)
            to_remove = []
            for cust in ungrouped:
                if dist_mat[seed][cust] < 0.2 * route_radial([seed]):
                    group.append(cust)
                    to_remove.append(cust)
            for cust in to_remove:
                ungrouped.remove(cust)
            customer_groups.append(group)
        # Sort groups by size (largest first)
        customer_groups.sort(key=lambda g: -len(g))
        insertion_order = []
        for group in customer_groups:
            group.sort(key=lambda c: demands[c], reverse=True)
            insertion_order.extend(group)
    else:
        insertion_order = list(removed_customers)
    
    # Step 2: Multi-criteria regret insertion
    uninserted = insertion_order[:]
    while uninserted:
        route_loads = [route_load(route) for route in new_x]
        avg_load = sum(route_loads) / len(route_loads) if route_loads else 0
        
        best_customer = None
        best_score = -float('inf')
        best_insert_info = None
        
        for customer in uninserted:
            cust_demand = demands[customer]
            feasible_options = []
            
            # Evaluate existing routes
            for r_idx, route in enumerate(new_x):
                if route_load(route) + cust_demand > capacity:
                    continue
                best_pos_cost = float('inf')
                best_pos = -1
                for pos in range(len(route) + 1):
                    cost = insertion_cost(route, pos, customer)
                    if cost < best_pos_cost:
                        best_pos_cost = cost
                        best_pos = pos
                if best_pos != -1:
                    attractiveness = route_attractiveness(route, customer)
                    load_balance = 1.0 - abs((route_load(route) + cust_demand - avg_load) / capacity)
                    combined_score = attractiveness * load_balance
                    fullness = (route_load(route) + cust_demand) / capacity
                    adjusted_cost = best_pos_cost * (1.0 - 0.3 * fullness)
                    feasible_options.append((adjusted_cost, combined_score, r_idx, best_pos))
            
            # New route option
            new_route_cost = insertion_cost([], 0, customer)
            feasible_options.append((new_route_cost, 1.0, len(new_x), 0))
            
            if not feasible_options:
                continue
            
            feasible_options.sort(key=lambda t: t[0])
            k = min(3, len(feasible_options))
            regret_sum = 0
            for i in range(1, k):
                regret_sum += feasible_options[i][0] - feasible_options[0][0]
            avg_regret = regret_sum / (k - 1) if k > 1 else 0
            
            customer_priority = 1.0
            if coords is not None and new_x:
                min_dist_to_any = min(
                    min(dist_mat[customer][node] for node in route if route)
                    for route in new_x if route
                ) if any(route for route in new_x) else 0
                customer_priority = 1.0 + min_dist_to_any / 100.0
            
            base_score = avg_regret * 0.5 + feasible_options[0][1] * 0.3 + customer_priority * 0.2
            feasible_count = len([opt for opt in feasible_options if opt[0] < float('inf')])
            demand_penalty = 1.0 - min(1.0, feasible_count / 5.0) if cust_demand > capacity * 0.3 else 0.0
            final_score = base_score * (1.0 - demand_penalty * 0.2)
            
            if final_score > best_score:
                best_score = final_score
                best_customer = customer
                best_insert_info = feasible_options[0]
        
        # Insert best customer
        if best_customer is not None:
            _, _, best_route_idx, best_pos = best_insert_info
            if best_route_idx == len(new_x):
                new_x.append([best_customer])
            else:
                new_x[best_route_idx].insert(best_pos, best_customer)
            uninserted.remove(best_customer)
        else:
            for customer in uninserted[:]:
                new_x.append([customer])
                uninserted.remove(customer)
    
    # Step 3: Local optimization of inserted customers (simplified)
    for route in new_x:
        if len(route) <= 2:
            continue
        improved = True
        while improved:
            improved = False
            for i in range(len(route)):
                customer = route[i]
                if customer not in removed_customers:
                    continue
                best_pos = i
                best_cost = float('inf')
                for j in range(len(route)):
                    temp_route = route[:i] + route[i+1:]
                    if j < len(temp_route):
                        temp_route.insert(j, customer)
                    else:
                        temp_route.append(customer)
                    # Compute total route cost
                    cost = dist_mat[depot][temp_route[0]] if temp_route else 0
                    for k in range(1, len(temp_route)):
                        cost += dist_mat[temp_route[k-1]][temp_route[k]]
                    if temp_route:
                        cost += dist_mat[temp_route[-1]][depot]
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = j
                if best_pos != i:
                    route.pop(i)
                    if best_pos < len(route):
                        route.insert(best_pos, customer)
                    else:
                        route.append(customer)
                    improved = True
                    break
    
    return new_x

# Score: 341.0
def repair_v4(x, removed_customers, problem_data):
    new_x = [route[:] for route in x]
    demands = problem_data['demands']
    capacity = problem_data['capacity']
    dist_mat = problem_data['distance_matrix']
    depot = problem_data.get('depot_idx', 0)
    coords = problem_data.get('coordinates', None)
    
    # Helper: route load
    def route_load(route):
        return sum(demands[node] for node in route)
    
    # Helper: route centroid distance
    def route_centroid_distance(route, customer):
        if not route:
            return dist_mat[depot][customer]
        avg_dist = sum(dist_mat[node][customer] for node in route) / len(route)
        return avg_dist
    
    # Helper: insertion cost with angular penalty
    def insertion_cost(route, pos, customer):
        if not route:
            return dist_mat[depot][customer] + dist_mat[customer][depot]
        
        prev_node = route[pos-1] if pos > 0 else depot
        next_node = route[pos] if pos < len(route) else depot
        base_cost = dist_mat[prev_node][customer] + dist_mat[customer][next_node] - dist_mat[prev_node][next_node]
        
        # Angular deviation penalty
        if coords is not None and prev_node != depot and next_node != depot:
            import math
            
            def vector_angle(x1, y1, x2, y2):
                return math.atan2(y2 - y1, x2 - x1)
            
            prev_coord = coords[prev_node]
            cust_coord = coords[customer]
            next_coord = coords[next_node]
            
            angle1 = vector_angle(prev_coord[0], prev_coord[1], cust_coord[0], cust_coord[1])
            angle2 = vector_angle(cust_coord[0], cust_coord[1], next_coord[0], next_coord[1])
            angle3 = vector_angle(prev_coord[0], prev_coord[1], next_coord[0], next_coord[1])
            
            angular_dev = abs(angle1 + angle2 - angle3)
            if angular_dev > math.pi:
                angular_dev = 2 * math.pi - angular_dev
            
            base_cost += angular_dev * 0.15 * (dist_mat[prev_node][customer] + dist_mat[customer][next_node])
        
        return base_cost
    
    # Helper: route compactness score
    def route_compactness(route):
        if len(route) <= 1:
            return 1.0
        
        route_dist = 0
        for i in range(len(route)-1):
            route_dist += dist_mat[route[i]][route[i+1]]
        
        if route_dist == 0:
            return 1.0
        
        # Calculate sum of distances from each node to route centroid
        centroid_dist = 0
        for node in route:
            min_dist = min(dist_mat[node][other] for other in route if other != node)
            centroid_dist += min_dist
        
        return route_dist / (centroid_dist + 1e-10)
    
    # Step 1: Adaptive customer prioritization
    if len(removed_customers) > 0:
        # Calculate customer centrality
        customer_scores = []
        for cust in removed_customers:
            # Distance to depot
            depot_dist = dist_mat[depot][cust]
            
            # Distance to other removed customers
            if len(removed_customers) > 1:
                other_dists = [dist_mat[cust][other] for other in removed_customers if other != cust]
                avg_other_dist = sum(other_dists) / len(other_dists) if other_dists else 0
            else:
                avg_other_dist = 0
            
            # Demand ratio
            demand_ratio = demands[cust] / capacity
            
            # Combined priority score (higher = insert later)
            priority = 0.4 * depot_dist + 0.3 * avg_other_dist + 0.3 * demand_ratio * 100
            customer_scores.append((cust, priority))
        
        # Sort by priority (highest first - insert difficult customers later)
        customer_scores.sort(key=lambda x: x[1], reverse=True)
        insertion_order = [cust for cust, _ in customer_scores]
    else:
        insertion_order = []
    
    # Step 2: Route fitness evaluation with adaptive weights
    def route_fitness(route, customer, current_loads, avg_load):
        load = route_load(route)
        spare_capacity = capacity - load - demands[customer]
        
        if spare_capacity < 0:
            return 0.0
        
        # Dynamic weight adjustment based on solution state
        total_customers = sum(len(r) for r in new_x) + len(insertion_order)
        routes_count = len(new_x)
        
        if routes_count == 0:
            load_weight, compact_weight, dist_weight = 0.3, 0.4, 0.3
        else:
            fullness_ratio = sum(current_loads) / (routes_count * capacity)
            if fullness_ratio > 0.8:
                load_weight, compact_weight, dist_weight = 0.5, 0.3, 0.2
            else:
                load_weight, compact_weight, dist_weight = 0.3, 0.4, 0.3
        
        # Load balance component
        projected_load = load + demands[customer]
        load_score = 1.0 - abs(projected_load - avg_load) / capacity
        
        # Compactness component
        compact_score = route_compactness(route + [customer]) if route else 1.0
        
        # Distance component
        if route:
            centroid_dist = route_centroid_distance(route, customer)
            max_possible = max(dist_mat[depot][node] for node in route) if route else 1
            dist_score = 1.0 - (centroid_dist / (max_possible + 1e-10))
        else:
            dist_score = 1.0
        
        return (load_weight * load_score + 
                compact_weight * compact_score + 
                dist_weight * dist_score)
    
    # Step 3: Predictive regret insertion with lookahead
    uninserted = insertion_order[:]
    
    while uninserted:
        # Calculate current statistics
        route_loads = [route_load(route) for route in new_x]
        avg_load = sum(route_loads) / len(route_loads) if route_loads else 0
        
        # Evaluate each customer
        customer_evaluations = []
        
        for customer in uninserted:
            cust_demand = demands[customer]
            insertion_options = []
            
            # Evaluate existing routes
            for r_idx, route in enumerate(new_x):
                if route_load(route) + cust_demand > capacity:
                    continue
                
                # Find best insertion position
                best_pos_cost = float('inf')
                best_pos = -1
                
                for pos in range(len(route) + 1):
                    cost = insertion_cost(route, pos, customer)
                    if cost < best_pos_cost:
                        best_pos_cost = cost
                        best_pos = pos
                
                if best_pos != -1:
                    fitness = route_fitness(route, customer, route_loads, avg_load)
                    # Adjust cost by fitness
                    adjusted_cost = best_pos_cost * (1.2 - fitness)  # Lower cost for higher fitness
                    insertion_options.append((adjusted_cost, fitness, r_idx, best_pos))
            
            # New route option
            new_route_cost = insertion_cost([], 0, customer)
            insertion_options.append((new_route_cost, 1.0, len(new_x), 0))
            
            if not insertion_options:
                customer_evaluations.append((customer, -float('inf'), None))
                continue
            
            # Sort by adjusted cost
            insertion_options.sort(key=lambda t: t[0])
            
            # Calculate predictive regret (consider impact on future insertions)
            if len(insertion_options) > 1:
                # Estimate how this insertion affects route attractiveness for remaining customers
                regret = insertion_options[1][0] - insertion_options[0][0]
                
                # Add penalty if this insertion consumes significant capacity
                capacity_usage = cust_demand / capacity
                if capacity_usage > 0.4:
                    regret *= (1.0 + capacity_usage)
                
                # Consider route count impact (penalize new route creation if many routes exist)
                if insertion_options[0][2] == len(new_x) and len(new_x) > 0:
                    avg_route_size = sum(len(r) for r in new_x) / len(new_x)
                    if avg_route_size < 5:  # Prefer to fill existing routes
                        regret *= 0.7
            else:
                regret = 0
            
            # Combine regret with fitness
            best_fitness = insertion_options[0][1]
            combined_score = regret * 0.6 + best_fitness * 0.4
            
            customer_evaluations.append((customer, combined_score, insertion_options[0]))
        
        # Select best customer to insert
        if customer_evaluations:
            customer_evaluations.sort(key=lambda x: x[1], reverse=True)
            best_customer, best_score, best_option = customer_evaluations[0]
            
            if best_option is not None:
                _, _, best_route_idx, best_pos = best_option
                if best_route_idx == len(new_x):
                    new_x.append([best_customer])
                else:
                    new_x[best_route_idx].insert(best_pos, best_customer)
                uninserted.remove(best_customer)
            else:
                # Fallback: insert customer with highest demand into new route
                if uninserted:
                    fallback_customer = max(uninserted, key=lambda c: demands[c])
                    new_x.append([fallback_customer])
                    uninserted.remove(fallback_customer)
        else:
            break
    
    # Step 4: Local refinement with simulated annealing approach
    for route_idx, route in enumerate(new_x):
        if len(route) <= 2:
            continue
        
        # Identify recently inserted customers in this route
        recent_customers = [cust for cust in route if cust in removed_customers]
        if not recent_customers:
            continue
        
        # Try to improve positions with limited neighborhood search
        for customer in recent_customers:
            if customer not in route:
                continue
            
            current_idx = route.index(customer)
            best_idx = current_idx
            best_route_cost = 0
            
            # Calculate current route cost
            for i in range(len(route)):
                if i == 0:
                    best_route_cost += dist_mat[depot][route[i]]
                else:
                    best_route_cost += dist_mat[route[i-1]][route[i]]
            if route:
                best_route_cost += dist_mat[route[-1]][depot]
            
            # Evaluate alternative positions (limited to ±3 positions)
            start_idx = max(0, current_idx - 3)
            end_idx = min(len(route), current_idx + 4)
            
            for new_idx in range(start_idx, end_idx):
                if new_idx == current_idx:
                    continue
                
                # Create candidate route
                temp_route = route[:]
                temp_route.pop(current_idx)
                if new_idx < len(temp_route):
                    temp_route.insert(new_idx, customer)
                else:
                    temp_route.append(customer)
                
                # Calculate new cost
                new_cost = 0
                for i in range(len(temp_route)):
                    if i == 0:
                        new_cost += dist_mat[depot][temp_route[i]]
                    else:
                        new_cost += dist_mat[temp_route[i-1]][temp_route[i]]
                if temp_route:
                    new_cost += dist_mat[temp_route[-1]][depot]
                
                if new_cost < best_route_cost:
                    best_route_cost = new_cost
                    best_idx = new_idx
            
            # Apply improvement if found
            if best_idx != current_idx:
                route.pop(current_idx)
                if best_idx < len(route):
                    route.insert(best_idx, customer)
                else:
                    route.append(customer)
    
    return new_x

# Score: 226.1
def repair_v5(x, removed_customers, problem_data):
    new_x = [route[:] for route in x]
    demands = problem_data['demands']
    capacity = problem_data['capacity']
    dist_mat = problem_data['distance_matrix']
    depot = problem_data.get('depot_idx', 0)
    
    # Helper functions
    def route_load(route):
        return sum(demands[c] for c in route)
    
    def insertion_cost(route, pos, customer):
        prev_node = route[pos-1] if pos > 0 else depot
        next_node = route[pos] if pos < len(route) else depot
        return dist_mat[prev_node][customer] + dist_mat[customer][next_node] - dist_mat[prev_node][next_node]
    
    def new_route_cost(customer):
        return dist_mat[depot][customer] + dist_mat[customer][depot]
    
    # Parameter settings
    ALPHA = 0.3  # Randomization factor (0 = pure greedy, 1 = fully random)
    BETA = 0.7   # Regret weight factor
    K_REGRET = 3 # Number of best positions to consider for regret
    EPSILON = 1e-6
    
    import random
    import math
    
    # Shuffle to randomize insertion order
    random.shuffle(removed_customers)
    
    for customer in removed_customers:
        cust_demand = demands[customer]
        
        # Collect feasible insertion options
        feasible_options = []
        
        # Option 1: Insert into existing routes
        for r_idx, route in enumerate(new_x):
            if route_load(route) + cust_demand > capacity:
                continue
            
            best_pos = -1
            best_cost = float('inf')
            cost_list = []
            
            for pos in range(len(route) + 1):
                cost = insertion_cost(route, pos, customer)
                cost_list.append((cost, pos))
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            
            if best_pos != -1:
                feasible_options.append(('existing', r_idx, best_pos, best_cost, cost_list))
        
        # Option 2: Create new route
        new_cost = new_route_cost(customer)
        feasible_options.append(('new', len(new_x), 0, new_cost, [(new_cost, 0)]))
        
        if not feasible_options:
            # Force create new route if no feasible options
            new_x.append([customer])
            continue
        
        # Apply regret-k insertion with randomization
        if random.random() < ALPHA:
            # Random selection among top candidates
            feasible_options.sort(key=lambda x: x[3])
            top_n = max(1, int(len(feasible_options) * 0.3))
            selected = random.choice(feasible_options[:top_n])
        else:
            # Regret-based selection
            if len(feasible_options) > 1:
                # Calculate regret values
                regret_values = []
                for opt_type, r_idx, pos, best_cost, cost_list in feasible_options:
                    if opt_type == 'new':
                        regret = 0
                    else:
                        # Get K_REGRET best alternative positions
                        cost_list.sort(key=lambda x: x[0])
                        alt_costs = [c for c, _ in cost_list[:K_REGRET]]
                        if len(alt_costs) > 1:
                            regret = sum(alt_costs[1:]) - (len(alt_costs)-1) * alt_costs[0]
                        else:
                            regret = 0
                    
                    # Combine greedy cost and regret
                    score = best_cost - BETA * regret
                    regret_values.append((score, (opt_type, r_idx, pos)))
                
                # Select with probability based on scores
                regret_values.sort(key=lambda x: x[0])
                min_score = regret_values[0][0]
                
                # Boltzmann selection
                temperatures = [math.exp(-(score - min_score) / (EPSILON + abs(min_score) * 0.1)) 
                              for score, _ in regret_values]
                total_temp = sum(temperatures)
                if total_temp > 0:
                    probs = [t/total_temp for t in temperatures]
                    selected_idx = random.choices(range(len(regret_values)), weights=probs)[0]
                    selected = feasible_options[selected_idx]
                else:
                    selected = feasible_options[0]
            else:
                selected = feasible_options[0]
        
        # Perform insertion
        opt_type, r_idx, pos, _, _ = selected
        
        if opt_type == 'new':
            if r_idx == len(new_x):
                new_x.append([customer])
            else:
                new_x[r_idx].insert(pos, customer)
        else:
            new_x[r_idx].insert(pos, customer)
    
    return new_x
