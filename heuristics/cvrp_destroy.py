# Best Solution: 7.84

# Score: 342.3
def destroy_v1(x, destroy_cnt, problem_data):
    import random
    import numpy as np
    
    dist_mat = problem_data.get('distance_matrix')
    if dist_mat is None:
        return destroy_v1(x, destroy_cnt, problem_data)
        
    new_x = [route[:] for route in x]
    removed_customers = []
    depot = problem_data.get('depot_idx', 0)
    
    # Adaptive parameter: blend between greedy and random
    greedy_ratio = 0.7  # 70% greedy, 30% random
    adaptive_threshold = 0.3  # For dynamic switching
    
    # Calculate savings for all nodes
    savings = []
    for r_idx, route in enumerate(new_x):
        for i, node in enumerate(route):
            prev_node = route[i-1] if i > 0 else depot
            next_node = route[i+1] if i < len(route)-1 else depot
            
            cost_with = dist_mat[prev_node][node] + dist_mat[node][next_node]
            cost_without = dist_mat[prev_node][next_node]
            
            saving = cost_with - cost_without
            # Add small random noise to break ties
            saving += random.uniform(-0.001, 0.001)
            savings.append((saving, node, r_idx, i))
    
    # Sort by saving (descending)
    savings.sort(key=lambda x: x[0], reverse=True)
    
    # Adaptive selection strategy
    if len(savings) > destroy_cnt * 2:
        # Use hybrid selection
        num_greedy = int(destroy_cnt * greedy_ratio)
        num_random = destroy_cnt - num_greedy
        
        # Greedy selection from top candidates
        top_candidates = savings[:min(len(savings), num_greedy * 3)]
        if random.random() < adaptive_threshold:
            # Occasionally shuffle top candidates for diversity
            random.shuffle(top_candidates)
        greedy_targets = top_candidates[:num_greedy]
        
        # Random selection from remaining nodes
        remaining = [s for s in savings if s[1] not in [t[1] for t in greedy_targets]]
        if remaining:
            random_targets = random.sample(remaining, min(num_random, len(remaining)))
            targets_info = greedy_targets + random_targets
        else:
            targets_info = greedy_targets
    else:
        # Not enough candidates, use pure greedy
        targets_info = savings[:destroy_cnt]
    
    # Remove customers in reverse order of position to maintain indices
    targets_info.sort(key=lambda x: (x[2], -x[3]))  # Sort by route idx, then reverse position
    
    for saving, customer, r_idx, pos_idx in targets_info:
        if customer not in removed_customers:
            removed_customers.append(customer)
            # Find and remove from the correct route
            for route in new_x:
                if customer in route:
                    route.remove(customer)
                    break
    
    # Clean up empty routes
    new_x = [r for r in new_x if len(r) > 0]
    
    # Ensure we removed exactly destroy_cnt customers (or as many as possible)
    if len(removed_customers) < destroy_cnt and len(removed_customers) < len(savings):
        # Additional random removal to reach target count
        all_customers = [node for route in x for node in route]
        remaining_customers = [c for c in all_customers if c not in removed_customers]
        additional_cnt = min(destroy_cnt - len(removed_customers), len(remaining_customers))
        if additional_cnt > 0:
            additional_removals = random.sample(remaining_customers, additional_cnt)
            removed_customers.extend(additional_removals)
            for customer in additional_removals:
                for route in new_x:
                    if customer in route:
                        route.remove(customer)
                        break
            new_x = [r for r in new_x if len(r) > 0]
    
    return removed_customers, new_x

# Score: 310.4
def destroy_v2(x, destroy_cnt, problem_data):
    import random
    import math
    import numpy as np
    
    def compute_route_metrics(route, dist_mat, depot):
        if len(route) == 0:
            return 0.0, 0.0, 0.0
        
        total_cost = 0.0
        current = depot
        for node in route:
            total_cost += dist_mat[current][node]
            current = node
        total_cost += dist_mat[current][depot]
        
        if len(route) <= 1:
            return total_cost, total_cost, 0.0
        
        max_saving = -float('inf')
        min_saving = float('inf')
        for idx, node in enumerate(route):
            prev = route[idx-1] if idx > 0 else depot
            nxt = route[idx+1] if idx < len(route)-1 else depot
            saving = dist_mat[prev][node] + dist_mat[node][nxt] - dist_mat[prev][nxt]
            max_saving = max(max_saving, saving)
            min_saving = min(min_saving, saving)
        
        saving_range = max_saving - min_saving if max_saving > min_saving else 1.0
        return total_cost, max_saving, saving_range
    
    def compute_adaptive_score(node, route_idx, route, dist_mat, coordinates, demands, depot, 
                              route_metrics, global_params):
        total_cost, max_saving, saving_range = route_metrics[route_idx]
        
        idx = route.index(node)
        prev = route[idx-1] if idx > 0 else depot
        nxt = route[idx+1] if idx < len(route)-1 else depot
        
        saving = dist_mat[prev][node] + dist_mat[node][nxt] - dist_mat[prev][nxt]
        normalized_saving = (saving - max_saving + saving_range) / saving_range if saving_range > 0 else 0.5
        
        route_coords = coordinates[route]
        centroid = np.mean(route_coords, axis=0)
        node_coord = coordinates[node]
        spatial_dist = np.linalg.norm(node_coord - centroid)
        
        route_demands = demands[route]
        total_demand = np.sum(route_demands)
        demand_ratio = demands[node] / total_demand if total_demand > 0 else 0
        
        route_len_penalty = len(route) / global_params['max_route_len'] if global_params['max_route_len'] > 0 else 1.0
        cost_weight = total_cost / global_params['avg_route_cost'] if global_params['avg_route_cost'] > 0 else 1.0
        
        spatial_component = spatial_dist * (1.0 + demand_ratio)
        saving_component = normalized_saving * (1.0 + route_len_penalty)
        
        adaptive_weight = 0.6 + 0.4 * math.exp(-global_params['iteration'] / 50.0)
        score = adaptive_weight * saving_component + (1.0 - adaptive_weight) * spatial_component
        
        noise = random.uniform(-0.0005 * score, 0.0005 * score) if score != 0 else random.uniform(-0.001, 0.001)
        return score + noise
    
    def compute_global_parameters(new_x, dist_mat, depot):
        if not new_x:
            return {'max_route_len': 1, 'avg_route_cost': 1.0, 'iteration': 0}
        
        route_lengths = [len(route) for route in new_x]
        route_costs = []
        
        for route in new_x:
            if len(route) == 0:
                continue
            cost = 0.0
            current = depot
            for node in route:
                cost += dist_mat[current][node]
                current = node
            cost += dist_mat[current][depot]
            route_costs.append(cost)
        
        max_route_len = max(route_lengths) if route_lengths else 1
        avg_route_cost = np.mean(route_costs) if route_costs else 1.0
        
        return {
            'max_route_len': max_route_len,
            'avg_route_cost': avg_route_cost,
            'iteration': random.randint(0, 1000)
        }
    
    dist_mat = problem_data.get('distance_matrix')
    coordinates = problem_data.get('coordinates')
    demands = problem_data.get('demands')
    capacity = problem_data.get('capacity')
    depot = problem_data.get('depot_idx', 0)
    
    if coordinates is not None and not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    if demands is not None and not isinstance(demands, np.ndarray):
        demands = np.array(demands)
    
    new_x = [route[:] for route in x]
    removed_customers = []
    
    all_customers = [c for route in new_x for c in route]
    if len(all_customers) <= destroy_cnt:
        return all_customers, [[]]
    
    if dist_mat is None or coordinates is None or demands is None:
        targets = random.sample(all_customers, destroy_cnt)
        for customer in targets:
            removed_customers.append(customer)
            for route in new_x:
                if customer in route:
                    route.remove(customer)
                    break
        new_x = [r for r in new_x if len(r) > 0]
        return removed_customers, new_x
    
    global_params = compute_global_parameters(new_x, dist_mat, depot)
    
    route_metrics = []
    for route in new_x:
        route_metrics.append(compute_route_metrics(route, dist_mat, depot))
    
    scores = []
    for r_idx, route in enumerate(new_x):
        if len(route) == 0:
            continue
        for node in route:
            score = compute_adaptive_score(node, r_idx, route, dist_mat, coordinates, 
                                          demands, depot, route_metrics, global_params)
            scores.append((score, node, r_idx))
    
    if not scores:
        targets = random.sample(all_customers, min(destroy_cnt, len(all_customers)))
        for customer in targets:
            removed_customers.append(customer)
            for route in new_x:
                if customer in route:
                    route.remove(customer)
                    break
        new_x = [r for r in new_x if len(r) > 0]
        return removed_customers, new_x
    
    scores.sort(key=lambda x: x[0], reverse=True)
    
    adaptive_greedy_ratio = 0.65 + 0.15 * math.sin(global_params['iteration'] / 100.0)
    num_greedy = int(destroy_cnt * adaptive_greedy_ratio)
    num_random = destroy_cnt - num_greedy
    
    if len(scores) > destroy_cnt * 2:
        candidate_pool_size = min(len(scores), num_greedy * 4)
        top_candidates = scores[:candidate_pool_size]
        
        if random.random() < 0.25:
            random.shuffle(top_candidates)
        
        greedy_targets = top_candidates[:num_greedy]
        
        remaining = [s for s in scores if s[1] not in [t[1] for t in greedy_targets]]
        if remaining and num_random > 0:
            random_targets = random.sample(remaining, min(num_random, len(remaining)))
            targets_info = greedy_targets + random_targets
        else:
            targets_info = greedy_targets
    else:
        targets_info = scores[:destroy_cnt]
    
    targets = [t[1] for t in targets_info]
    removal_order = sorted(targets_info, key=lambda x: x[0], reverse=True)
    
    for _, customer, _ in removal_order:
        removed_customers.append(customer)
        for route in new_x:
            if customer in route:
                route.remove(customer)
                break
    
    new_x = [r for r in new_x if len(r) > 0]
    
    if len(removed_customers) < destroy_cnt and len(removed_customers) < len(all_customers):
        remaining_customers = [c for c in all_customers if c not in removed_customers]
        additional_cnt = min(destroy_cnt - len(removed_customers), len(remaining_customers))
        if additional_cnt > 0:
            additional_removals = random.sample(remaining_customers, additional_cnt)
            removed_customers.extend(additional_removals)
            for customer in additional_removals:
                for route in new_x:
                    if customer in route:
                        route.remove(customer)
                        break
            new_x = [r for r in new_x if len(r) > 0]
    
    return removed_customers, new_x

# Score: 341.0
def destroy_v3(x, destroy_cnt, problem_data):
    import random
    import math
    
    def compute_savings(dist_mat, depot, route, node):
        idx = route.index(node)
        prev = route[idx-1] if idx > 0 else depot
        nxt = route[idx+1] if idx < len(route)-1 else depot
        cost_with = dist_mat[prev][node] + dist_mat[node][nxt]
        cost_without = dist_mat[prev][nxt]
        return cost_with - cost_without
    
    def compute_spatial_score(coords, demands, route, node, total_demand):
        route_coords = [coords[c] for c in route]
        centroid_x = sum(c[0] for c in route_coords) / len(route)
        centroid_y = sum(c[1] for c in route_coords) / len(route)
        node_coord = coords[node]
        dist = math.sqrt((node_coord[0] - centroid_x)**2 + (node_coord[1] - centroid_y)**2)
        demand_ratio = demands[node] / total_demand if total_demand > 0 else 0
        return dist * (1.0 - demand_ratio)
    
    dist_mat = problem_data.get('distance_matrix')
    coordinates = problem_data.get('coordinates')
    demands = problem_data.get('demands')
    capacity = problem_data.get('capacity')
    depot = problem_data.get('depot_idx', 0)
    
    new_x = [route[:] for route in x]
    removed_customers = []
    
    all_customers = [c for route in new_x for c in route]
    if len(all_customers) <= destroy_cnt:
        return all_customers, [[]]
    
    if dist_mat is None or coordinates is None or demands is None:
        targets = random.sample(all_customers, destroy_cnt)
        for customer in targets:
            removed_customers.append(customer)
            for route in new_x:
                if customer in route:
                    route.remove(customer)
                    break
        new_x = [r for r in new_x if len(r) > 0]
        return removed_customers, new_x
    
    scores = []
    for r_idx, route in enumerate(new_x):
        if len(route) == 0:
            continue
        total_demand = sum(demands[c] for c in route)
        for node in route:
            saving = compute_savings(dist_mat, depot, route, node)
            spatial = compute_spatial_score(coordinates, demands, route, node, total_demand)
            combined = saving + spatial
            scores.append((combined, node, r_idx))
    
    if not scores:
        targets = random.sample(all_customers, min(destroy_cnt, len(all_customers)))
        for customer in targets:
            removed_customers.append(customer)
            for route in new_x:
                if customer in route:
                    route.remove(customer)
                    break
        new_x = [r for r in new_x if len(r) > 0]
        return removed_customers, new_x
    
    scores.sort(key=lambda x: x[0], reverse=True)
    destroy_targets = min(destroy_cnt, len(scores))
    targets_info = scores[:destroy_targets]
    targets = [t[1] for t in targets_info]
    
    for customer in targets:
        removed_customers.append(customer)
        for route in new_x:
            if customer in route:
                route.remove(customer)
                break
    
    new_x = [r for r in new_x if len(r) > 0]
    return removed_customers, new_x

# Score: 292.6
def destroy_v4(x, destroy_cnt, problem_data):
    import random
    import math
    import numpy as np
    
    def compute_route_metrics(route, dist_mat, depot):
        if len(route) == 0:
            return 0.0, 0.0, 0.0
        
        total_cost = 0.0
        current = depot
        for node in route:
            total_cost += dist_mat[current][node]
            current = node
        total_cost += dist_mat[current][depot]
        
        if len(route) <= 1:
            return total_cost, total_cost, 0.0
        
        max_saving = -float('inf')
        min_saving = float('inf')
        for idx, node in enumerate(route):
            prev = route[idx-1] if idx > 0 else depot
            nxt = route[idx+1] if idx < len(route)-1 else depot
            saving = dist_mat[prev][node] + dist_mat[node][nxt] - dist_mat[prev][nxt]
            max_saving = max(max_saving, saving)
            min_saving = min(min_saving, saving)
        
        saving_range = max_saving - min_saving if max_saving > min_saving else 1.0
        return total_cost, max_saving, saving_range
    
    def compute_adaptive_score(node, route_idx, route, dist_mat, coordinates, demands, depot, 
                              route_metrics, global_params):
        total_cost, max_saving, saving_range = route_metrics[route_idx]
        
        idx = route.index(node)
        prev = route[idx-1] if idx > 0 else depot
        nxt = route[idx+1] if idx < len(route)-1 else depot
        
        saving = dist_mat[prev][node] + dist_mat[node][nxt] - dist_mat[prev][nxt]
        normalized_saving = (saving - max_saving + saving_range) / saving_range if saving_range > 0 else 0.5
        
        route_coords = coordinates[route]
        centroid = np.mean(route_coords, axis=0)
        node_coord = coordinates[node]
        spatial_dist = np.linalg.norm(node_coord - centroid)
        
        route_demands = demands[route]
        total_demand = np.sum(route_demands)
        demand_ratio = demands[node] / total_demand if total_demand > 0 else 0
        
        route_len_penalty = len(route) / global_params['max_route_len'] if global_params['max_route_len'] > 0 else 1.0
        cost_weight = total_cost / global_params['avg_route_cost'] if global_params['avg_route_cost'] > 0 else 1.0
        
        spatial_component = spatial_dist * (1.0 + demand_ratio)
        saving_component = normalized_saving * (1.0 + route_len_penalty)
        
        adaptive_weight = 0.6 + 0.4 * math.exp(-global_params['iteration'] / 50.0)
        score = adaptive_weight * saving_component + (1.0 - adaptive_weight) * spatial_component
        
        noise = random.uniform(-0.0005 * score, 0.0005 * score) if score != 0 else random.uniform(-0.001, 0.001)
        return score + noise
    
    def compute_global_parameters(new_x, dist_mat, depot):
        if not new_x:
            return {'max_route_len': 1, 'avg_route_cost': 1.0, 'iteration': 0}
        
        route_lengths = [len(route) for route in new_x]
        route_costs = []
        
        for route in new_x:
            if len(route) == 0:
                continue
            cost = 0.0
            current = depot
            for node in route:
                cost += dist_mat[current][node]
                current = node
            cost += dist_mat[current][depot]
            route_costs.append(cost)
        
        max_route_len = max(route_lengths) if route_lengths else 1
        avg_route_cost = np.mean(route_costs) if route_costs else 1.0
        
        return {
            'max_route_len': max_route_len,
            'avg_route_cost': avg_route_cost,
            'iteration': random.randint(0, 1000)
        }
    
    dist_mat = problem_data.get('distance_matrix')
    coordinates = problem_data.get('coordinates')
    demands = problem_data.get('demands')
    capacity = problem_data.get('capacity')
    depot = problem_data.get('depot_idx', 0)
    
    if coordinates is not None and not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    if demands is not None and not isinstance(demands, np.ndarray):
        demands = np.array(demands)
    
    new_x = [route[:] for route in x]
    removed_customers = []
    
    all_customers = [c for route in new_x for c in route]
    if len(all_customers) <= destroy_cnt:
        return all_customers, [[]]
    
    if dist_mat is None or coordinates is None or demands is None:
        targets = random.sample(all_customers, destroy_cnt)
        for customer in targets:
            removed_customers.append(customer)
            for route in new_x:
                if customer in route:
                    route.remove(customer)
                    break
        new_x = [r for r in new_x if len(r) > 0]
        return removed_customers, new_x
    
    global_params = compute_global_parameters(new_x, dist_mat, depot)
    
    route_metrics = []
    for route in new_x:
        route_metrics.append(compute_route_metrics(route, dist_mat, depot))
    
    scores = []
    for r_idx, route in enumerate(new_x):
        if len(route) == 0:
            continue
        for node in route:
            score = compute_adaptive_score(node, r_idx, route, dist_mat, coordinates, 
                                          demands, depot, route_metrics, global_params)
            scores.append((score, node, r_idx))
    
    if not scores:
        targets = random.sample(all_customers, min(destroy_cnt, len(all_customers)))
        for customer in targets:
            removed_customers.append(customer)
            for route in new_x:
                if customer in route:
                    route.remove(customer)
                    break
        new_x = [r for r in new_x if len(r) > 0]
        return removed_customers, new_x
    
    scores.sort(key=lambda x: x[0], reverse=True)
    
    adaptive_greedy_ratio = 0.65 + 0.15 * math.sin(global_params['iteration'] / 100.0)
    num_greedy = int(destroy_cnt * adaptive_greedy_ratio)
    num_random = destroy_cnt - num_greedy
    
    if len(scores) > destroy_cnt * 2:
        candidate_pool_size = min(len(scores), num_greedy * 4)
        top_candidates = scores[:candidate_pool_size]
        
        if random.random() < 0.25:
            random.shuffle(top_candidates)
        
        greedy_targets = top_candidates[:num_greedy]
        
        remaining = [s for s in scores if s[1] not in [t[1] for t in greedy_targets]]
        if remaining and num_random > 0:
            random_targets = random.sample(remaining, min(num_random, len(remaining)))
            targets_info = greedy_targets + random_targets
        else:
            targets_info = greedy_targets
    else:
        targets_info = scores[:destroy_cnt]
    
    targets = [t[1] for t in targets_info]
    removal_order = sorted(targets_info, key=lambda x: x[0], reverse=True)
    
    for _, customer, _ in removal_order:
        removed_customers.append(customer)
        for route in new_x:
            if customer in route:
                route.remove(customer)
                break
    
    new_x = [r for r in new_x if len(r) > 0]
    
    if len(removed_customers) < destroy_cnt and len(removed_customers) < len(all_customers):
        remaining_customers = [c for c in all_customers if c not in removed_customers]
        additional_cnt = min(destroy_cnt - len(removed_customers), len(remaining_customers))
        if additional_cnt > 0:
            additional_removals = random.sample(remaining_customers, additional_cnt)
            removed_customers.extend(additional_removals)
            for customer in additional_removals:
                for route in new_x:
                    if customer in route:
                        route.remove(customer)
                        break
            new_x = [r for r in new_x if len(r) > 0]
    
    return removed_customers, new_x

# Score: 327.2
def destroy_v5(x, destroy_cnt, problem_data):
    import random
    import numpy as np
    
    dist_mat = problem_data.get('distance_matrix')
    if dist_mat is None:
        return destroy_v1(x, destroy_cnt, problem_data)
        
    new_x = [route[:] for route in x]
    removed_customers = []
    depot = problem_data.get('depot_idx', 0)
    
    # Adaptive parameters with dynamic adjustment
    base_greedy_ratio = 0.75
    adaptive_threshold = 0.25
    noise_intensity = 0.005
    
    # Calculate removal scores for all nodes
    removal_scores = []
    for r_idx, route in enumerate(new_x):
        if len(route) == 0:
            continue
            
        for i, node in enumerate(route):
            # Get neighbors
            prev_node = route[i-1] if i > 0 else depot
            next_node = route[i+1] if i < len(route)-1 else depot
            
            # Calculate removal impact
            current_cost = dist_mat[prev_node][node] + dist_mat[node][next_node]
            new_cost = dist_mat[prev_node][next_node]
            cost_impact = current_cost - new_cost
            
            # Consider route characteristics
            route_density = len(route)
            position_factor = min(i, len(route)-1-i) / max(1, len(route)-1)
            
            # Composite score favoring high-impact, isolated nodes
            score = cost_impact * (1 + 0.2 * position_factor)
            
            # Add adaptive noise for exploration
            noise = random.uniform(-noise_intensity * abs(score), noise_intensity * abs(score))
            score += noise
            
            removal_scores.append((score, node, r_idx, i, cost_impact))
    
    if not removal_scores:
        return [], new_x
    
    # Sort by score (descending - higher score means better removal candidate)
    removal_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Dynamic greedy ratio based on search state
    total_nodes = sum(len(route) for route in x)
    if total_nodes > 0:
        removal_ratio = destroy_cnt / total_nodes
        # Increase randomness for larger removals
        adjusted_greedy_ratio = base_greedy_ratio * (1 - 0.3 * removal_ratio)
    else:
        adjusted_greedy_ratio = base_greedy_ratio
    
    # Hybrid selection strategy
    if len(removal_scores) > destroy_cnt:
        num_greedy = int(destroy_cnt * adjusted_greedy_ratio)
        num_random = destroy_cnt - num_greedy
        
        # Greedy selection with occasional diversification
        if random.random() < adaptive_threshold:
            # Explore medium-impact candidates
            start_idx = random.randint(0, min(3, len(removal_scores) - num_greedy))
            greedy_candidates = removal_scores[start_idx:start_idx + num_greedy * 2]
        else:
            # Standard greedy from top
            greedy_candidates = removal_scores[:num_greedy * 2]
        
        # Ensure we have enough candidates
        if len(greedy_candidates) < num_greedy:
            greedy_candidates = removal_scores[:num_greedy]
        
        # Select greedy removals
        selected_greedy = []
        for candidate in greedy_candidates:
            if len(selected_greedy) >= num_greedy:
                break
            if candidate[1] not in removed_customers:
                selected_greedy.append(candidate)
        
        # Random selection from remaining nodes
        remaining = [s for s in removal_scores if s[1] not in [t[1] for t in selected_greedy]]
        if remaining and num_random > 0:
            # Weight random selection by inverse score to avoid very good nodes
            weights = [1.0 / (abs(s[0]) + 1.0) for s in remaining]
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w/total_weight for w in weights]
                random_indices = np.random.choice(len(remaining), 
                                                 size=min(num_random, len(remaining)), 
                                                 replace=False, 
                                                 p=weights)
                selected_random = [remaining[i] for i in random_indices]
            else:
                selected_random = random.sample(remaining, min(num_random, len(remaining)))
        else:
            selected_random = []
        
        targets_info = selected_greedy + selected_random
    else:
        # Not enough candidates, use all available
        targets_info = removal_scores[:destroy_cnt]
    
    # Remove customers in reverse order to maintain indices
    targets_info.sort(key=lambda x: (x[2], -x[3]))
    
    for score, customer, r_idx, pos_idx, cost_impact in targets_info:
        if customer not in removed_customers:
            removed_customers.append(customer)
            # Find and remove from the correct route
            for route in new_x:
                if customer in route:
                    route.remove(customer)
                    break
    
    # Clean up empty routes
    new_x = [r for r in new_x if len(r) > 0]
    
    # Ensure we removed exactly destroy_cnt customers
    if len(removed_customers) < destroy_cnt:
        all_customers = [node for route in x for node in route]
        remaining_customers = [c for c in all_customers if c not in removed_customers]
        
        if remaining_customers:
            additional_cnt = min(destroy_cnt - len(removed_customers), len(remaining_customers))
            
            # Cluster-based additional removal for efficiency
            if additional_cnt > 1 and len(remaining_customers) > additional_cnt:
                # Group by proximity to already removed customers
                candidate_scores = []
                for customer in remaining_customers:
                    # Find minimum distance to already removed customers
                    if removed_customers:
                        min_dist = min(dist_mat[customer][rc] for rc in removed_customers)
                        candidate_scores.append((min_dist, customer))
                    else:
                        candidate_scores.append((0, customer))
                
                # Prefer customers close to already removed ones (cluster removal)
                candidate_scores.sort(key=lambda x: x[0])
                additional_removals = [c for _, c in candidate_scores[:additional_cnt]]
            else:
                additional_removals = random.sample(remaining_customers, additional_cnt)
            
            removed_customers.extend(additional_removals)
            for customer in additional_removals:
                for route in new_x:
                    if customer in route:
                        route.remove(customer)
                        break
            
            new_x = [r for r in new_x if len(r) > 0]
    
    return removed_customers, new_x
