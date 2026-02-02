import random
# Best Solution: 5.69

# Score: 299.2
def destroy_v1(x, destroy_cnt, problem_data):
    """Cluster-based destroy operator with adaptive randomization for OVRP"""
    
    # Helper functions defined inside
    def calculate_route_cost(route):
        """Calculate total distance of a route"""
        if not route:
            return 0
            
        total = 0
        prev_node = problem_data['depot_idx']
        for customer in route:
            total += problem_data['distance_matrix'][prev_node][customer]
            prev_node = customer
        return total
    
    def find_clusters(customers, k):
        """Group customers into k clusters using farthest-first traversal"""
        if len(customers) <= k:
            return [[c] for c in customers]
        
        # Start with a random customer
        clusters = []
        remaining = set(customers)
        
        # Pick first center randomly
        first_center = random.choice(list(remaining))
        clusters.append([first_center])
        remaining.remove(first_center)
        
        # Pick remaining centers using farthest-first
        centers = [first_center]
        while len(clusters) < k and remaining:
            # Find customer farthest from all current centers
            max_dist = -1
            farthest_customer = None
            
            for customer in remaining:
                min_dist_to_centers = min(
                    problem_data['distance_matrix'][customer][center] 
                    for center in centers
                )
                if min_dist_to_centers > max_dist:
                    max_dist = min_dist_to_centers
                    farthest_customer = customer
            
            if farthest_customer:
                clusters.append([farthest_customer])
                centers.append(farthest_customer)
                remaining.remove(farthest_customer)
        
        # Assign remaining customers to nearest cluster
        for customer in remaining:
            # Find nearest cluster center
            min_dist = float('inf')
            best_cluster_idx = 0
            
            for i, center in enumerate(centers):
                dist = problem_data['distance_matrix'][customer][center]
                if dist < min_dist:
                    min_dist = dist
                    best_cluster_idx = i
            
            clusters[best_cluster_idx].append(customer)
        
        return clusters
    
    def calculate_customer_removal_cost(customer, route, route_idx):
        """Calculate cost savings from removing a customer"""
        if not route:
            return 0
            
        try:
            pos = route.index(customer)
        except ValueError:
            return 0
            
        route_len = len(route)
        depot_idx = problem_data['depot_idx']
        
        if route_len == 1:
            # Only customer in route
            return problem_data['distance_matrix'][depot_idx][customer]
        
        if pos == 0:
            # First customer
            next_customer = route[1]
            current_cost = (problem_data['distance_matrix'][depot_idx][customer] + 
                          problem_data['distance_matrix'][customer][next_customer])
            removed_cost = problem_data['distance_matrix'][depot_idx][next_customer]
        elif pos == route_len - 1:
            # Last customer
            prev_customer = route[pos-1]
            current_cost = problem_data['distance_matrix'][prev_customer][customer]
            removed_cost = 0  # No connection after last customer
        else:
            # Middle customer
            prev_customer = route[pos-1]
            next_customer = route[pos+1]
            current_cost = (problem_data['distance_matrix'][prev_customer][customer] + 
                          problem_data['distance_matrix'][customer][next_customer])
            removed_cost = problem_data['distance_matrix'][prev_customer][next_customer]
        
        return current_cost - removed_cost
    
    # Main function logic
    new_x = [route[:] for route in x]  # Deep copy
    removed_customers = []
    
    # Extract all customers
    all_customers = [c for route in new_x for c in route]
    
    # Edge cases
    if not all_customers or destroy_cnt == 0:
        return [], new_x
    
    if len(all_customers) <= destroy_cnt:
        # Remove all customers
        removed_customers = all_customers[:]
        new_x = [[] for _ in new_x]
        return removed_customers, new_x
    
    # Build customer to route mapping once
    customer_to_route = {}
    customer_to_position = {}
    for route_idx, route in enumerate(new_x):
        for pos, customer in enumerate(route):
            customer_to_route[customer] = route_idx
            customer_to_position[customer] = pos
    
    # Adaptive strategy selection
    total_customers = len(all_customers)
    
    # Use cluster-based removal for larger destroy counts, worst removal for smaller
    use_cluster_removal = (destroy_cnt >= total_customers * 0.3)
    
    if use_cluster_removal:
        # Cluster-based removal: remove entire clusters of nearby customers
        # This creates larger "holes" in the solution for repair to fill
        
        # Determine number of clusters based on destroy count
        num_clusters = min(destroy_cnt, max(2, destroy_cnt // 3))
        
        # Create clusters of customers
        clusters = find_clusters(all_customers, num_clusters)
        
        # Sort clusters by size (largest first)
        clusters.sort(key=len, reverse=True)
        
        # Select clusters to remove
        selected_customers = set()
        for cluster in clusters:
            if len(selected_customers) >= destroy_cnt:
                break
            
            # Calculate average removal cost for this cluster
            cluster_removal_costs = []
            for customer in cluster:
                if customer in customer_to_route:
                    route_idx = customer_to_route[customer]
                    cost = calculate_customer_removal_cost(
                        customer, new_x[route_idx], route_idx
                    )
                    cluster_removal_costs.append((cost, customer))
            
            if not cluster_removal_costs:
                continue
            
            # Sort by removal cost (highest first)
            cluster_removal_costs.sort(reverse=True, key=lambda x: x[0])
            
            # Take customers from this cluster until we reach destroy_cnt
            for cost, customer in cluster_removal_costs:
                if len(selected_customers) >= destroy_cnt:
                    break
                if customer not in selected_customers:
                    selected_customers.add(customer)
        
        # If we still need more customers, add random ones
        if len(selected_customers) < destroy_cnt:
            remaining = [c for c in all_customers if c not in selected_customers]
            needed = destroy_cnt - len(selected_customers)
            if remaining and needed > 0:
                additional = random.sample(remaining, min(needed, len(remaining)))
                selected_customers.update(additional)
        
        removed_customers = list(selected_customers)
        
    else:
        # Worst removal with adaptive randomization
        # Calculate removal cost for each customer
        removal_costs = []
        for customer in all_customers:
            route_idx = customer_to_route[customer]
            cost = calculate_customer_removal_cost(
                customer, new_x[route_idx], route_idx
            )
            removal_costs.append((cost, customer))
        
        # Sort by cost (highest first)
        removal_costs.sort(reverse=True, key=lambda x: x[0])
        
        # Adaptive randomization parameter
        # More randomization for smaller destroy counts
        p = max(1, 10 - destroy_cnt // 5)
        
        # Select customers with weighted probability
        candidates_to_consider = min(len(removal_costs), max(destroy_cnt * 2, 10))
        candidate_customers = [rc[1] for rc in removal_costs[:candidates_to_consider]]
        
        # Create weighted probabilities (exponential decay)
        weights = [(candidates_to_consider - i) ** p for i in range(candidates_to_consider)]
        total_weight = sum(weights)
        if total_weight > 0:
            probabilities = [w / total_weight for w in weights]
        else:
            probabilities = [1.0 / candidates_to_consider] * candidates_to_consider
        
        # Select customers
        selected_customers = set()
        attempts = 0
        max_attempts = destroy_cnt * 3
        
        while len(selected_customers) < destroy_cnt and attempts < max_attempts:
            if not candidate_customers:
                break
                
            idx = random.choices(
                range(len(candidate_customers)), 
                weights=probabilities, 
                k=1
            )[0]
            customer = candidate_customers[idx]
            
            if customer not in selected_customers:
                selected_customers.add(customer)
            
            attempts += 1
        
        # Fallback if we didn't get enough
        if len(selected_customers) < destroy_cnt:
            remaining_needed = destroy_cnt - len(selected_customers)
            for cost, customer in removal_costs:
                if len(selected_customers) >= destroy_cnt:
                    break
                if customer not in selected_customers:
                    selected_customers.add(customer)
        
        removed_customers = list(selected_customers)
    
    # Remove selected customers efficiently
    # Group removals by route to minimize list modifications
    removals_by_route = {}
    for customer in removed_customers:
        if customer in customer_to_route:
            route_idx = customer_to_route[customer]
            if route_idx not in removals_by_route:
                removals_by_route[route_idx] = []
            removals_by_route[route_idx].append(customer)
    
    # Remove customers from routes
    for route_idx, customers_to_remove in removals_by_route.items():
        if route_idx < len(new_x):
            # Remove in reverse order to maintain indices
            customers_to_remove.sort(
                key=lambda c: customer_to_position[c], 
                reverse=True
            )
            for customer in customers_to_remove:
                if customer in new_x[route_idx]:
                    new_x[route_idx].remove(customer)
    
    # Clean up empty routes
    new_x = [route for route in new_x if route]
    
    return removed_customers, new_x

# Score: 337.0
def destroy_v2(x, destroy_cnt, problem_data):
    """Improved destroy operator using worst removal with randomization and better edge case handling"""
    # x is list of routes (lists of ints)
    new_x = [route[:] for route in x]
    removed_customers = []
    
    # Extract problem data
    distance_matrix = problem_data['distance_matrix']
    demands = problem_data['demands']
    capacity = problem_data['capacity']
    depot_idx = problem_data['depot_idx']
    
    # Flatten all customers in current solution
    all_customers = [c for route in new_x for c in route]
    
    if len(all_customers) <= destroy_cnt:
        # If we're removing all or more customers, just return all
        return all_customers, [[]]
    
    # Early return for edge case
    if destroy_cnt == 0:
        return [], new_x
    
    # Dynamic strategy selection based on problem size and destroy count
    total_customers = len(all_customers)
    use_worst_removal = (destroy_cnt < total_customers * 0.7 and destroy_cnt > 0)
    
    if use_worst_removal:
        # Pre-allocate list for cost savings
        cost_savings = []
        
        # Build a mapping for faster removal later
        customer_to_route_idx = {}
        
        for route_idx, route in enumerate(new_x):
            route_len = len(route)
            if route_len == 0:
                continue
                
            # Store mapping for all customers in this route
            for customer in route:
                customer_to_route_idx[customer] = route_idx
            
            # Calculate cost savings for each customer in the route
            for i, customer in enumerate(route):
                # Determine previous node
                prev_node = depot_idx if i == 0 else route[i-1]
                
                if i == route_len - 1:
                    # Last customer: only cost from previous to customer
                    cost = distance_matrix[prev_node][customer]
                else:
                    # Middle customer: calculate savings from bypassing
                    next_node = route[i+1]
                    current_cost = distance_matrix[prev_node][customer] + distance_matrix[customer][next_node]
                    removed_cost = distance_matrix[prev_node][next_node]
                    cost = current_cost - removed_cost
                
                cost_savings.append((cost, customer, route_idx))
        
        # Early return if no cost savings calculated (shouldn't happen but safe)
        if not cost_savings:
            # Fall back to random removal
            targets = random.sample(all_customers, min(destroy_cnt, len(all_customers)))
            removed_customers = targets[:]
            
            # Remove customers using the mapping we built
            for customer in targets:
                route_idx = customer_to_route_idx.get(customer)
                if route_idx is not None and customer in new_x[route_idx]:
                    new_x[route_idx].remove(customer)
        else:
            # Sort by cost savings (highest first)
            cost_savings.sort(reverse=True, key=lambda x: x[0])
            
            # Adaptive candidate selection
            p = 6  # Randomization parameter
            candidates_to_consider = min(len(cost_savings), max(destroy_cnt * 3, 10))
            
            # Create weighted probability distribution
            indices = list(range(candidates_to_consider))
            
            # Use exponential decay for probabilities
            probabilities = [(candidates_to_consider - i) ** p for i in indices]
            total_prob = sum(probabilities)
            
            # Normalize probabilities
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
            else:
                # Fallback to uniform distribution
                probabilities = [1.0 / candidates_to_consider] * candidates_to_consider
            
            # Select customers with weighted randomness
            selected_customers = set()
            attempts = 0
            max_attempts = destroy_cnt * 2
            
            while len(selected_customers) < destroy_cnt and attempts < max_attempts:
                if candidates_to_consider <= 0:
                    break
                    
                idx = random.choices(indices, weights=probabilities, k=1)[0]
                customer = cost_savings[idx][1]
                
                if customer not in selected_customers:
                    selected_customers.add(customer)
                
                attempts += 1
            
            # If we didn't get enough unique customers, fill with worst ones
            if len(selected_customers) < destroy_cnt:
                remaining_needed = destroy_cnt - len(selected_customers)
                for i in range(candidates_to_consider):
                    if len(selected_customers) >= destroy_cnt:
                        break
                    customer = cost_savings[i][1]
                    if customer not in selected_customers:
                        selected_customers.add(customer)
            
            # Remove selected customers using the pre-built mapping
            for customer in selected_customers:
                removed_customers.append(customer)
                route_idx = customer_to_route_idx.get(customer)
                if route_idx is not None and customer in new_x[route_idx]:
                    new_x[route_idx].remove(customer)
            
            # Final safety check: if still not enough, use random removal
            if len(removed_customers) < destroy_cnt:
                remaining_needed = destroy_cnt - len(removed_customers)
                current_customers = [c for route in new_x for c in route]
                if current_customers and remaining_needed > 0:
                    additional = random.sample(
                        current_customers, 
                        min(remaining_needed, len(current_customers))
                    )
                    for customer in additional:
                        if customer not in removed_customers:
                            removed_customers.append(customer)
                            route_idx = customer_to_route_idx.get(customer)
                            if route_idx is not None and customer in new_x[route_idx]:
                                new_x[route_idx].remove(customer)
    else:
        # Use random removal with optimized approach
        targets = random.sample(all_customers, destroy_cnt)
        removed_customers = targets[:]
        
        # Build mapping once
        customer_to_route_idx = {}
        for route_idx, route in enumerate(new_x):
            for customer in route:
                customer_to_route_idx[customer] = route_idx
        
        # Remove customers using the mapping
        for customer in targets:
            route_idx = customer_to_route_idx.get(customer)
            if route_idx is not None and customer in new_x[route_idx]:
                new_x[route_idx].remove(customer)
    
    # Filter empty routes efficiently
    new_x = [r for r in new_x if r]  # Using truthiness check for empty lists
    
    return removed_customers, new_x

# Score: 317.4
def destroy_v3(x, destroy_cnt, problem_data):
    """Enhanced destroy operator with route-based and proximity-based removal strategies"""
    
    # Helper functions
    def calculate_route_demand(route):
        """Calculate total demand of a route"""
        return sum(problem_data['demands'][c] for c in route)
    
    def calculate_route_distance(route):
        """Calculate total distance of a route"""
        if not route:
            return 0
            
        total = problem_data['distance_matrix'][0][route[0]]  # Depot to first
        for i in range(len(route) - 1):
            total += problem_data['distance_matrix'][route[i]][route[i+1]]
        return total
    
    def find_related_customers(customer, all_customers, n=5):
        """Find n customers closest to the given customer"""
        distances = []
        for other in all_customers:
            if other != customer:
                dist = problem_data['distance_matrix'][customer][other]
                distances.append((dist, other))
        
        distances.sort(key=lambda x: x[0])
        return [other for _, other in distances[:min(n, len(distances))]]
    
    def calculate_removal_impact(customer, route):
        """Calculate the impact of removing a customer from its route"""
        if not route or customer not in route:
            return 0
        
        idx = route.index(customer)
        route_len = len(route)
        
        if route_len == 1:
            # Removing the only customer
            return problem_data['distance_matrix'][0][customer]
        
        if idx == 0:
            # First customer
            old_dist = (problem_data['distance_matrix'][0][customer] +
                       problem_data['distance_matrix'][customer][route[1]])
            new_dist = problem_data['distance_matrix'][0][route[1]]
        elif idx == route_len - 1:
            # Last customer
            old_dist = problem_data['distance_matrix'][route[idx-1]][customer]
            new_dist = 0  # Route ends at previous customer
        else:
            # Middle customer
            old_dist = (problem_data['distance_matrix'][route[idx-1]][customer] +
                       problem_data['distance_matrix'][customer][route[idx+1]])
            new_dist = problem_data['distance_matrix'][route[idx-1]][route[idx+1]]
        
        return old_dist - new_dist
    
    # Main function
    new_x = [route[:] for route in x]  # Deep copy
    removed_customers = []
    
    # Extract all customers
    all_customers = [c for route in new_x for c in route]
    
    # Edge cases
    if not all_customers or destroy_cnt == 0:
        return [], new_x
    
    if len(all_customers) <= destroy_cnt:
        # Remove all customers
        removed_customers = all_customers[:]
        new_x = [[] for _ in new_x]
        return removed_customers, new_x
    
    # Build customer to route mapping
    customer_to_route = {}
    for route_idx, route in enumerate(new_x):
        for customer in route:
            customer_to_route[customer] = route_idx
    
    # Adaptive strategy selection with more randomization
    strategies = ['route_based', 'proximity_based', 'worst_removal', 'random']
    strategy_weights = [0.4, 0.3, 0.2, 0.1]
    
    # Randomly select strategy with given weights
    selected_strategy = random.choices(strategies, weights=strategy_weights, k=1)[0]
    
    if selected_strategy == 'route_based':
        # Remove entire routes or parts of routes
        # Focus on routes that are inefficient or have slack capacity
        
        # Calculate route metrics
        route_metrics = []
        for route_idx, route in enumerate(new_x):
            if not route:
                continue
                
            route_dist = calculate_route_distance(route)
            route_demand = calculate_route_demand(route)
            utilization = route_demand / problem_data['capacity']
            
            # Score routes: lower utilization and higher distance per customer are worse
            if len(route) > 0:
                dist_per_customer = route_dist / len(route)
                # Higher score means worse route
                score = (1 - utilization) * 0.4 + dist_per_customer * 0.6
                route_metrics.append((score, route_idx, route))
        
        # Sort by score (worst first)
        route_metrics.sort(reverse=True, key=lambda x: x[0])
        
        # Select customers from worst routes
        selected_customers = set()
        for score, route_idx, route in route_metrics:
            if len(selected_customers) >= destroy_cnt:
                break
            
            # Determine how many to remove from this route (at least 1, at most all)
            max_from_route = min(len(route), destroy_cnt - len(selected_customers))
            if max_from_route <= 0:
                continue
            
            # Remove a portion of the route (30-70%)
            remove_count = max(1, min(max_from_route, random.randint(
                max(1, len(route) // 3), 
                max(2, len(route) * 2 // 3)
            )))
            
            # Select which customers to remove from this route
            if random.random() < 0.5:
                # Remove consecutive segment
                start = random.randint(0, len(route) - remove_count)
                to_remove = route[start:start + remove_count]
            else:
                # Remove random customers from route
                to_remove = random.sample(route, min(remove_count, len(route)))
            
            selected_customers.update(to_remove)
    
    elif selected_strategy == 'proximity_based':
        # Remove clusters of nearby customers
        # Start with a random customer and remove its neighbors
        
        selected_customers = set()
        
        while len(selected_customers) < destroy_cnt:
            # Pick a random seed customer not yet removed
            available = [c for c in all_customers if c not in selected_customers]
            if not available:
                break
                
            seed = random.choice(available)
            selected_customers.add(seed)
            
            # Find related customers
            related = find_related_customers(seed, available, n=5)
            
            # Remove some of the related customers
            if related:
                num_to_remove = min(
                    random.randint(1, 3),
                    destroy_cnt - len(selected_customers),
                    len(related)
                )
                to_remove = random.sample(related, num_to_remove)
                selected_customers.update(to_remove)
    
    elif selected_strategy == 'worst_removal':
        # Remove customers with highest removal impact
        removal_impacts = []
        for customer in all_customers:
            route_idx = customer_to_route[customer]
            impact = calculate_removal_impact(customer, new_x[route_idx])
            removal_impacts.append((impact, customer))
        
        # Sort by impact (highest first)
        removal_impacts.sort(reverse=True, key=lambda x: x[0])
        
        # Select top customers with some randomization
        top_n = min(len(removal_impacts), destroy_cnt * 2)
        top_candidates = [c for _, c in removal_impacts[:top_n]]
        
        # Use roulette wheel selection based on impact rank
        selected_customers = set()
        while len(selected_customers) < destroy_cnt and top_candidates:
            # Create weights (higher impact = higher weight)
            weights = [top_n - i for i in range(len(top_candidates))]
            total_weight = sum(weights)
            
            if total_weight > 0:
                idx = random.choices(range(len(top_candidates)), weights=weights, k=1)[0]
                customer = top_candidates[idx]
                selected_customers.add(customer)
                
                # Remove from candidates
                top_candidates.pop(idx)
            else:
                break
    
    else:  # 'random'
        # Pure random removal
        selected_customers = set(random.sample(all_customers, min(destroy_cnt, len(all_customers))))
    
    # Ensure we have exactly destroy_cnt customers
    if len(selected_customers) > destroy_cnt:
        # Randomly select subset
        selected_customers = set(random.sample(list(selected_customers), destroy_cnt))
    elif len(selected_customers) < destroy_cnt:
        # Add random customers to reach destroy_cnt
        remaining = [c for c in all_customers if c not in selected_customers]
        needed = destroy_cnt - len(selected_customers)
        if remaining and needed > 0:
            additional = random.sample(remaining, min(needed, len(remaining)))
            selected_customers.update(additional)
    
    removed_customers = list(selected_customers)
    
    # Efficient removal by rebuilding routes
    new_routes = []
    for route in new_x:
        new_route = [c for c in route if c not in selected_customers]
        if new_route:  # Only keep non-empty routes
            new_routes.append(new_route)
    
    return removed_customers, new_routes

# Score: 279.9
def destroy_v4(x, destroy_cnt, problem_data):
    import random
    import math
    
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
    
    new_x = [route[:] for route in x]
    removed_customers = []
    all_customers = [c for route in new_x for c in route]
    
    if len(all_customers) <= destroy_cnt:
        return all_customers, [[]]
    
    coordinates = problem_data.get('coordinates')
    distance_matrix = problem_data.get('distance_matrix')
    depot_idx = problem_data.get('depot_idx', 0)
    
    if coordinates is None or distance_matrix is None:
        return destroy_v1(x, destroy_cnt, problem_data)
    
    customer_coords = {}
    for cust in all_customers:
        if cust < len(coordinates):
            customer_coords[cust] = coordinates[cust]
    
    if len(customer_coords) < destroy_cnt:
        return destroy_v1(x, destroy_cnt, problem_data)
    
    num_clusters = min(5, max(2, int(math.sqrt(len(all_customers)))))
    if len(customer_coords) < num_clusters:
        cluster_centers = list(customer_coords.keys())
    else:
        cluster_centers = random.sample(list(customer_coords.keys()), num_clusters)
    
    clusters = {center: [] for center in cluster_centers}
    
    for cust, coord in customer_coords.items():
        min_dist = float('inf')
        nearest_center = None
        for center in cluster_centers:
            center_coord = coordinates[center] if center < len(coordinates) else [0.5, 0.5]
            dx = coord[0] - center_coord[0]
            dy = coord[1] - center_coord[1]
            dist = dx*dx + dy*dy
            if dist < min_dist:
                min_dist = dist
                nearest_center = center
        if nearest_center is not None:
            clusters[nearest_center].append(cust)
    
    non_empty_clusters = {k: v for k, v in clusters.items() if len(v) > 0}
    
    if not non_empty_clusters:
        return destroy_v1(x, destroy_cnt, problem_data)
    
    cluster_savings = {}
    for center, customers in non_empty_clusters.items():
        savings_list = []
        for cust in customers:
            for r_idx, route in enumerate(new_x):
                if cust in route:
                    i = route.index(cust)
                    prev_node = route[i-1] if i > 0 else depot_idx
                    is_last = (i == len(route) - 1)
                    next_node = route[i+1] if not is_last else None
                    
                    if is_last:
                        saving = distance_matrix[prev_node][cust]
                    else:
                        cost_with = distance_matrix[prev_node][cust] + distance_matrix[cust][next_node]
                        cost_without = distance_matrix[prev_node][next_node]
                        saving = cost_with - cost_without
                    
                    savings_list.append((saving, cust))
                    break
        savings_list.sort(key=lambda x: x[0], reverse=True)
        cluster_savings[center] = savings_list
    
    targets = []
    remaining_to_remove = destroy_cnt
    
    while remaining_to_remove > 0 and any(len(savings) > 0 for savings in cluster_savings.values()):
        for center in list(cluster_savings.keys()):
            if remaining_to_remove <= 0:
                break
            if cluster_savings[center]:
                best_cust = cluster_savings[center][0][1]
                if best_cust not in targets:
                    targets.append(best_cust)
                    remaining_to_remove -= 1
                cluster_savings[center].pop(0)
    
    if len(targets) < destroy_cnt:
        remaining_customers = [c for c in all_customers if c not in targets]
        additional_needed = destroy_cnt - len(targets)
        if additional_needed <= len(remaining_customers):
            additional_targets = random.sample(remaining_customers, additional_needed)
            targets.extend(additional_targets)
        else:
            targets.extend(remaining_customers)
    
    for customer in targets:
        removed_customers.append(customer)
        for route in new_x:
            if customer in route:
                route.remove(customer)
                break
    
    new_x = [r for r in new_x if len(r) > 0]
    return removed_customers, new_x

# Score: 277.1
def destroy_v5(x, destroy_cnt, problem_data):
    new_x = [route[:] for route in x]
    removed_customers = []
    
    all_customers = [c for route in new_x for c in route]
    
    if len(all_customers) <= destroy_cnt:
        return all_customers, [[]]
    
    coordinates = problem_data.get('coordinates')
    
    depot_idx = problem_data.get('depot_idx', 0)
    customer_coords = {}
    for cust in all_customers:
        if cust < len(coordinates):
            customer_coords[cust] = coordinates[cust]
    
    if len(customer_coords) < destroy_cnt:
        targets = random.sample(all_customers, destroy_cnt)
        for customer in targets:
            removed_customers.append(customer)
            for route in new_x:
                if customer in route:
                    route.remove(customer)
                    break
        new_x = [r for r in new_x if len(r) > 0]
        return removed_customers, new_x
    
    cluster_centers = random.sample(list(customer_coords.keys()), min(5, len(customer_coords)))
    
    clusters = {center: [] for center in cluster_centers}
    
    for cust, coord in customer_coords.items():
        min_dist = float('inf')
        nearest_center = None
        for center in cluster_centers:
            center_coord = coordinates[center] if center < len(coordinates) else [0.5, 0.5]
            dx = coord[0] - center_coord[0]
            dy = coord[1] - center_coord[1]
            dist = dx*dx + dy*dy
            if dist < min_dist:
                min_dist = dist
                nearest_center = center
        if nearest_center is not None:
            clusters[nearest_center].append(cust)
    
    non_empty_clusters = {k: v for k, v in clusters.items() if len(v) > 0}
    
    if not non_empty_clusters:
        targets = random.sample(all_customers, destroy_cnt)
    else:
        cluster_list = list(non_empty_clusters.values())
        weights = [len(cluster) for cluster in cluster_list]
        total_weight = sum(weights)
        
        if total_weight == 0:
            targets = random.sample(all_customers, destroy_cnt)
        else:
            targets = []
            while len(targets) < destroy_cnt:
                chosen_cluster = random.choices(cluster_list, weights=weights, k=1)[0]
                if chosen_cluster:
                    cust = random.choice(chosen_cluster)
                    if cust not in targets:
                        targets.append(cust)
    
    for customer in targets:
        removed_customers.append(customer)
        for route in new_x:
            if customer in route:
                route.remove(customer)
                break
    
    new_x = [r for r in new_x if len(r) > 0]
    return removed_customers, new_x

