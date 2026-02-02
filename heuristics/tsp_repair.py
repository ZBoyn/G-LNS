import copy
import numpy as np
import random
import math

# Best Solution: 5.71
def insert_v1(x, removed_cities, problem_data):
    if problem_data is None:
        dist_mat = None
    elif isinstance(problem_data, dict):
        dist_mat = problem_data.get('distance_matrix')
    else:
        dist_mat = problem_data
        
    if dist_mat is None:
        # Fallback to v1 if no distance matrix
        def insert_v1(x, removed_cities, problem_data):
            new_x = copy.deepcopy(x)
            for city in removed_cities:
                best_pos = -1
                best_increase = float('inf')
                for j in range(len(new_x) + 1):
                    if j == 0:
                        prev = new_x[-1] if new_x else city
                        nxt = new_x[0] if new_x else city
                    elif j == len(new_x):
                        prev = new_x[-1]
                        nxt = new_x[0]
                    else:
                        prev = new_x[j-1]
                        nxt = new_x[j]
                    increase = dist_mat[prev][city] + dist_mat[city][nxt] - dist_mat[prev][nxt]
                    if increase < best_increase:
                        best_increase = increase
                        best_pos = j
                new_x.insert(best_pos, city)
            return new_x
        return insert_v1(x, removed_cities, problem_data)
    
    # Helper functions defined inside main function
    def _calc_insertion_cost(prev_city, city, next_city):
        return dist_mat[prev_city][city] + dist_mat[city][next_city] - dist_mat[prev_city][next_city]
    
    def _tour_distance(path):
        if len(path) <= 1:
            return 0
        d = 0
        for i in range(len(path) - 1):
            d += dist_mat[path[i]][path[i + 1]]
        d += dist_mat[path[-1]][path[0]]
        return d
    
    def _greedy_insertion_with_randomization(candidate_cities, current_path):
        """Greedy insertion with adaptive randomization"""
        nonlocal randomization_factor, temperature
        
        # Adaptive parameter adjustment based on remaining cities
        remaining_ratio = len(candidate_cities) / (len(candidate_cities) + len(current_path))
        
        # Increase randomization when fewer cities remain (fine-tuning phase)
        adaptive_randomization = randomization_factor * (1.0 + 2.0 * (1.0 - remaining_ratio))
        
        # Dynamic temperature for simulated annealing-like acceptance
        adaptive_temperature = temperature * remaining_ratio
        
        path = copy.deepcopy(current_path)
        
        while candidate_cities:
            # Candidate selection with randomization
            if len(candidate_cities) > 1 and random.random() < adaptive_randomization:
                # Select random subset for evaluation
                k = min(3, len(candidate_cities))
                candidates = random.sample(candidate_cities, k)
            else:
                candidates = candidate_cities
            
            best_city = None
            best_pos = -1
            best_cost = float('inf')
            second_best_cost = float('inf')
            
            # Evaluate each candidate city
            for city in candidates:
                # Find best position for this city
                city_best_pos = -1
                city_best_cost = float('inf')
                
                for j in range(len(path) + 1):
                    if j == 0:
                        prev = path[-1] if path else city
                        nxt = path[0] if path else city
                    elif j == len(path):
                        prev = path[-1]
                        nxt = path[0]
                    else:
                        prev = path[j-1]
                        nxt = path[j]
                    
                    cost = _calc_insertion_cost(prev, city, nxt)
                    if cost < city_best_cost:
                        city_best_cost = cost
                        city_best_pos = j
                
                # Track best and second best
                if city_best_cost < best_cost:
                    second_best_cost = best_cost
                    best_cost = city_best_cost
                    best_pos = city_best_pos
                    best_city = city
                elif city_best_cost < second_best_cost:
                    second_best_cost = city_best_cost
            
            # Apply probabilistic acceptance of second best
            if (second_best_cost < float('inf') and 
                random.random() < adaptive_randomization and
                adaptive_temperature > 0):
                cost_diff = second_best_cost - best_cost
                acceptance_prob = np.exp(-cost_diff / adaptive_temperature)
                if random.random() < acceptance_prob:
                    # Re-evaluate for second best option
                    for city in candidates:
                        if city == best_city:
                            continue
                        city_best_pos = -1
                        city_best_cost = float('inf')
                        
                        for j in range(len(path) + 1):
                            if j == 0:
                                prev = path[-1] if path else city
                                nxt = path[0] if path else city
                            elif j == len(path):
                                prev = path[-1]
                                nxt = path[0]
                            else:
                                prev = path[j-1]
                                nxt = path[j]
                            
                            cost = _calc_insertion_cost(prev, city, nxt)
                            if cost < city_best_cost:
                                city_best_cost = cost
                                city_best_pos = j
                        
                        if abs(city_best_cost - second_best_cost) < 1e-6:
                            best_city = city
                            best_pos = city_best_pos
                            best_cost = city_best_cost
                            break
            
            # Insert the selected city
            path.insert(best_pos, best_city)
            candidate_cities.remove(best_city)
            
            # Cool down temperature
            adaptive_temperature *= 0.95
        
        return path
    
    # Parameter initialization with adaptive tuning
    randomization_factor = 0.15  # Base randomization level
    temperature = np.mean(dist_mat) * 0.1  # Dynamic temperature based on problem scale
    
    # Shuffle removed cities to avoid insertion order bias
    shuffled_cities = copy.deepcopy(removed_cities)
    random.shuffle(shuffled_cities)
    
    # Apply greedy insertion with adaptive randomization
    result = _greedy_insertion_with_randomization(shuffled_cities, x)
    
    # Local improvement: 2-opt on recently inserted segments
    if len(removed_cities) > 0 and len(result) > 3:
        # Identify segments containing inserted cities
        inserted_set = set(removed_cities)
        for _ in range(2):  # Limited iterations for speed
            improved = False
            for i in range(len(result)):
                if result[i] in inserted_set or result[(i+1)%len(result)] in inserted_set:
                    for j in range(i+2, len(result) + (i if i>0 else -1)):
                        j_mod = j % len(result)
                        if j_mod == i:
                            continue
                        if result[j_mod] in inserted_set or result[(j_mod+1)%len(result)] in inserted_set:
                            # Check if 2-opt swap improves
                            a, b = result[i], result[(i+1)%len(result)]
                            c, d = result[j_mod], result[(j_mod+1)%len(result)]
                            
                            old_cost = dist_mat[a][b] + dist_mat[c][d]
                            new_cost = dist_mat[a][c] + dist_mat[b][d]
                            
                            if new_cost < old_cost - 1e-6:
                                # Perform swap
                                if i < j_mod:
                                    result[i+1:j_mod+1] = reversed(result[i+1:j_mod+1])
                                else:
                                    segment = result[i+1:] + result[:j_mod+1]
                                    segment.reverse()
                                    result[i+1:] = segment[:len(result)-i-1]
                                    result[:j_mod+1] = segment[len(result)-i-1:]
                                improved = True
                                break
                    if improved:
                        break
            if not improved:
                break
    
    return result

def insert_v2(x, removed_cities, problem_data):
    if problem_data is None:
        dist_mat = None
    elif isinstance(problem_data, dict):
        dist_mat = problem_data.get('distance_matrix')
    else:
        dist_mat = problem_data
        
    if dist_mat is None:
        def insert_v1(x, removed_cities, problem_data):
            new_x = copy.deepcopy(x)
            for city in removed_cities:
                best_pos = -1
                best_increase = float('inf')
                for j in range(len(new_x) + 1):
                    if j == 0:
                        prev = new_x[-1] if new_x else city
                        nxt = new_x[0] if new_x else city
                    elif j == len(new_x):
                        prev = new_x[-1]
                        nxt = new_x[0]
                    else:
                        prev = new_x[j-1]
                        nxt = new_x[j]
                    increase = dist_mat[prev][city] + dist_mat[city][nxt] - dist_mat[prev][nxt]
                    if increase < best_increase:
                        best_increase = increase
                        best_pos = j
                new_x.insert(best_pos, city)
            return new_x
        return insert_v1(x, removed_cities, problem_data)
    
    def _calc_insertion_cost(prev_city, city, next_city):
        return dist_mat[prev_city][city] + dist_mat[city][next_city] - dist_mat[prev_city][next_city]
    
    def _compute_regret_values(cities, current_path):
        regret_data = []
        n = len(current_path)
        
        for city in cities:
            best_cost = float('inf')
            second_best_cost = float('inf')
            best_pos = -1
            
            for j in range(n + 1):
                if n == 0:
                    prev = city
                    nxt = city
                elif j == 0:
                    prev = current_path[-1]
                    nxt = current_path[0]
                elif j == n:
                    prev = current_path[-1]
                    nxt = current_path[0]
                else:
                    prev = current_path[j-1]
                    nxt = current_path[j]
                
                cost = _calc_insertion_cost(prev, city, nxt)
                if cost < best_cost:
                    second_best_cost = best_cost
                    best_cost = cost
                    best_pos = j
                elif cost < second_best_cost:
                    second_best_cost = cost
            
            regret = second_best_cost - best_cost if second_best_cost < float('inf') else 0
            normalized_regret = regret + best_cost * 0.001
            regret_data.append((city, best_cost, normalized_regret, best_pos))
        
        return regret_data
    
    def _adaptive_hybrid_insertion(candidate_cities, current_path):
        path = copy.deepcopy(current_path)
        remaining = copy.deepcopy(candidate_cities)
        
        n_total = len(path) + len(remaining)
        remaining_ratio = len(remaining) / n_total
        
        base_exploration = 0.4 * (1.0 - remaining_ratio) + 0.1
        regret_weight = 2.0 - 1.5 * remaining_ratio
        base_k = max(2, min(5, int(3 + 2 * remaining_ratio)))
        
        temperature = np.mean(dist_mat) * 0.1
        randomization_factor = 0.15
        
        iteration = 0
        while remaining:
            iteration += 1
            
            remaining_ratio_current = len(remaining) / n_total
            adaptive_randomization = randomization_factor * (1.0 + 2.0 * (1.0 - remaining_ratio_current))
            adaptive_temperature = temperature * remaining_ratio_current
            current_exploration = base_exploration * (0.95 ** iteration)
            
            regret_info = _compute_regret_values(remaining, path)
            
            if len(remaining) > 1 and random.random() < current_exploration:
                k = min(base_k, len(regret_info))
                
                weighted_scores = []
                for city, best_cost, regret, pos in regret_info:
                    score = regret_weight * regret - best_cost
                    weighted_scores.append((score, city, best_cost, pos))
                
                weighted_scores.sort(reverse=True)
                top_candidates = weighted_scores[:k]
                
                if len(top_candidates) > 1 and random.random() < adaptive_randomization:
                    scores = [c[0] for c in top_candidates]
                    max_score = max(scores)
                    min_score = min(scores)
                    
                    if max_score > min_score:
                        normalized = [(s - min_score) / (max_score - min_score + 1e-6) for s in scores]
                        probabilities = [n / sum(normalized) for n in normalized]
                        selected_idx = random.choices(range(len(top_candidates)), weights=probabilities)[0]
                    else:
                        selected_idx = random.randint(0, len(top_candidates)-1)
                else:
                    selected_idx = 0
                    
                selected = top_candidates[selected_idx]
                city_to_insert = selected[1]
                insert_pos = selected[3]
                best_cost = selected[2]
                
                if len(remaining) > 1 and random.random() < adaptive_randomization and adaptive_temperature > 0:
                    second_best_city = None
                    second_best_pos = -1
                    second_best_cost = float('inf')
                    
                    for city, cost, regret, pos in regret_info:
                        if city == city_to_insert:
                            continue
                        if cost < second_best_cost:
                            second_best_cost = cost
                            second_best_city = city
                            second_best_pos = pos
                    
                    if second_best_city is not None:
                        cost_diff = second_best_cost - best_cost
                        acceptance_prob = np.exp(-cost_diff / adaptive_temperature)
                        if random.random() < acceptance_prob:
                            city_to_insert = second_best_city
                            insert_pos = second_best_pos
            else:
                max_regret = -float('inf')
                city_to_insert = None
                insert_pos = -1
                min_best_cost = float('inf')
                
                for city, best_cost, regret, pos in regret_info:
                    if regret > max_regret:
                        max_regret = regret
                        city_to_insert = city
                        insert_pos = pos
                        min_best_cost = best_cost
                    elif abs(regret - max_regret) < 1e-6 and best_cost < min_best_cost:
                        city_to_insert = city
                        insert_pos = pos
                        min_best_cost = best_cost
            
            path.insert(insert_pos, city_to_insert)
            remaining.remove(city_to_insert)
            
            temperature *= 0.95
        
        return path
    
    def _segment_optimization(tour, inserted_set):
        improved = True
        iteration = 0
        n = len(tour)
        max_iterations = min(4, max(2, n // 20))
        
        while improved and iteration < max_iterations:
            improved = False
            base_neighborhood = min(4, max(2, n // 15))
            
            for neighborhood_size in [base_neighborhood, base_neighborhood + 1, base_neighborhood - 1]:
                if neighborhood_size < 2:
                    continue
                    
                inserted_positions = [i for i in range(n) if tour[i] in inserted_set]
                
                for i in inserted_positions:
                    start = max(0, i - neighborhood_size)
                    end = min(n, i + neighborhood_size + 1)
                    
                    segment = tour[start:end]
                    if len(segment) <= 3:
                        continue
                    
                    prev_city = tour[start-1] if start > 0 else tour[-1]
                    next_city = tour[end] if end < n else tour[0]
                    
                    current_cost = dist_mat[prev_city][segment[0]]
                    for idx in range(len(segment)-1):
                        current_cost += dist_mat[segment[idx]][segment[idx+1]]
                    current_cost += dist_mat[segment[-1]][next_city]
                    
                    best_segment = segment
                    best_cost = current_cost
                    
                    for s in range(len(segment)-1):
                        for e in range(s+2, len(segment)):
                            new_seg = segment.copy()
                            new_seg[s:e] = reversed(new_seg[s:e])
                            
                            cost = dist_mat[prev_city][new_seg[0]]
                            for idx in range(len(new_seg)-1):
                                cost += dist_mat[new_seg[idx]][new_seg[idx+1]]
                            cost += dist_mat[new_seg[-1]][next_city]
                            
                            if cost < best_cost - 1e-6:
                                best_cost = cost
                                best_segment = new_seg
                                improved = True
                    
                    if len(segment) <= 5 and not improved:
                        for rot in range(1, len(segment)):
                            new_seg = segment[rot:] + segment[:rot]
                            cost = dist_mat[prev_city][new_seg[0]]
                            for idx in range(len(new_seg)-1):
                                cost += dist_mat[new_seg[idx]][new_seg[idx+1]]
                            cost += dist_mat[new_seg[-1]][next_city]
                            if cost < best_cost - 1e-6:
                                best_cost = cost
                                best_segment = new_seg
                                improved = True
                    
                    if improved:
                        tour[start:end] = best_segment
                        break
                
                if improved:
                    break
            
            iteration += 1
        
        return tour
    
    def _local_refinement(tour):
        n = len(tour)
        improved = True
        max_refinement_iterations = 3
        
        for _ in range(max_refinement_iterations):
            if not improved:
                break
            improved = False
            
            for i in range(n-1):
                for j in range(i+2, n):
                    if j == n-1 and i == 0:
                        continue
                    
                    a, b = tour[i], tour[(i+1) % n]
                    c, d = tour[j], tour[(j+1) % n]
                    
                    current = dist_mat[a][b] + dist_mat[c][d]
                    new = dist_mat[a][c] + dist_mat[b][d]
                    
                    if new < current - 1e-6:
                        if j > i+1:
                            tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break
        
        return tour
    
    if not removed_cities:
        return copy.deepcopy(x)
    
    shuffled_cities = copy.deepcopy(removed_cities)
    random.shuffle(shuffled_cities)
    result = _adaptive_hybrid_insertion(shuffled_cities, x)
    
    if len(removed_cities) > 0 and len(result) > 3:
        inserted_set = set(removed_cities)
        result = _segment_optimization(result, inserted_set)
    
    if len(result) > 3:
        result = _local_refinement(result)
    
    return result

def insert_v3(x, removed_cities, problem_data):
    if problem_data is None:
        dist_mat = None
    elif isinstance(problem_data, dict):
        dist_mat = problem_data.get('distance_matrix')
    else:
        dist_mat = problem_data
        
    if dist_mat is None:
        def insert_v1(x, removed_cities, problem_data):
            new_x = copy.deepcopy(x)
            for city in removed_cities:
                best_pos = -1
                best_increase = float('inf')
                for j in range(len(new_x) + 1):
                    if j == 0:
                        prev = new_x[-1] if new_x else city
                        nxt = new_x[0] if new_x else city
                    elif j == len(new_x):
                        prev = new_x[-1]
                        nxt = new_x[0]
                    else:
                        prev = new_x[j-1]
                        nxt = new_x[j]
                    increase = dist_mat[prev][city] + dist_mat[city][nxt] - dist_mat[prev][nxt]
                    if increase < best_increase:
                        best_increase = increase
                        best_pos = j
                new_x.insert(best_pos, city)
            return new_x
        return insert_v1(x, removed_cities, problem_data)
    
    def _calc_insertion_cost(prev_city, city, next_city):
        return dist_mat[prev_city][city] + dist_mat[city][next_city] - dist_mat[prev_city][next_city]
    
    def _tour_distance(path):
        if len(path) <= 1:
            return 0
        d = 0
        for i in range(len(path) - 1):
            d += dist_mat[path[i]][path[i + 1]]
        d += dist_mat[path[-1]][path[0]]
        return d
    
    def _compute_regret_values(cities, current_path):
        regret_data = []
        for city in cities:
            best_cost = float('inf')
            second_best_cost = float('inf')
            best_pos = -1
            
            for j in range(len(current_path) + 1):
                if j == 0:
                    prev = current_path[-1] if current_path else city
                    nxt = current_path[0] if current_path else city
                elif j == len(current_path):
                    prev = current_path[-1]
                    nxt = current_path[0]
                else:
                    prev = current_path[j-1]
                    nxt = current_path[j]
                
                cost = _calc_insertion_cost(prev, city, nxt)
                if cost < best_cost:
                    second_best_cost = best_cost
                    best_cost = cost
                    best_pos = j
                elif cost < second_best_cost:
                    second_best_cost = cost
            
            regret = second_best_cost - best_cost if second_best_cost < float('inf') else 0
            regret_data.append((city, best_cost, regret, best_pos))
        
        return regret_data
    
    def _adaptive_regret_randomized_insertion(candidate_cities, current_path):
        path = copy.deepcopy(current_path)
        remaining = copy.deepcopy(candidate_cities)
        
        n_total = len(path) + len(remaining)
        base_exploration = 0.2 + 0.3 * (len(remaining) / n_total)
        regret_weight = 1.5 - 0.5 * (len(remaining) / n_total)
        temperature = np.mean(dist_mat) * 0.1 * (len(remaining) / n_total)
        
        while remaining:
            regret_info = _compute_regret_values(remaining, path)
            
            if len(remaining) > 1 and random.random() < base_exploration:
                k = min(3, len(regret_info))
                weighted_scores = []
                for city, best_cost, regret, pos in regret_info:
                    score = regret_weight * regret - best_cost
                    weighted_scores.append((score, city, best_cost, pos))
                
                weighted_scores.sort(reverse=True)
                candidates = weighted_scores[:k]
                
                if random.random() < 0.3 and temperature > 0:
                    cost_diffs = [c[2] for c in candidates]
                    probs = np.exp(np.array(cost_diffs) / temperature)
                    probs = probs / probs.sum()
                    idx = np.random.choice(len(candidates), p=probs)
                else:
                    idx = random.randrange(len(candidates))
                
                selected = candidates[idx]
                city_to_insert = selected[1]
                insert_pos = selected[3]
            else:
                max_regret = -float('inf')
                city_to_insert = None
                insert_pos = -1
                best_cost_for_max_regret = float('inf')
                
                for city, best_cost, regret, pos in regret_info:
                    if regret > max_regret or (abs(regret - max_regret) < 1e-6 and best_cost < best_cost_for_max_regret):
                        max_regret = regret
                        city_to_insert = city
                        insert_pos = pos
                        best_cost_for_max_regret = best_cost
            
            path.insert(insert_pos, city_to_insert)
            remaining.remove(city_to_insert)
            
            base_exploration *= 0.95
            temperature *= 0.95
        
        return path
    
    def _segment_optimization_vns(tour, inserted_set):
        improved = True
        iteration = 0
        
        while improved and iteration < 3:
            improved = False
            
            for neighborhood_size in [2, 3, 2]:
                for i in range(len(tour)):
                    if tour[i] not in inserted_set:
                        continue
                    
                    start = max(0, i - neighborhood_size)
                    end = min(len(tour), i + neighborhood_size + 1)
                    segment = tour[start:end]
                    
                    if len(segment) <= 3:
                        continue
                    
                    best_segment = segment
                    best_cost = _segment_cost(segment, tour[start-1] if start > 0 else tour[-1],
                                            tour[end] if end < len(tour) else tour[0])
                    
                    candidates = [segment]
                    
                    if len(segment) <= 5:
                        for s in range(len(segment)-1):
                            for e in range(s+2, len(segment)+1):
                                new_seg = segment.copy()
                                new_seg[s:e] = reversed(new_seg[s:e])
                                candidates.append(new_seg)
                    else:
                        for s in range(len(segment)-1):
                            for e in range(s+2, len(segment)):
                                new_seg = segment.copy()
                                new_seg[s:e] = reversed(new_seg[s:e])
                                candidates.append(new_seg)
                    
                    for candidate in candidates:
                        cost = _segment_cost(candidate, tour[start-1] if start > 0 else tour[-1],
                                           tour[end] if end < len(tour) else tour[0])
                        if cost < best_cost - 1e-6:
                            best_cost = cost
                            best_segment = candidate
                            improved = True
                    
                    if improved:
                        tour[start:end] = best_segment
                        break
                
                if improved:
                    break
            
            iteration += 1
        
        return tour
    
    def _segment_cost(segment, prev_city, next_city):
        if len(segment) == 0:
            return dist_mat[prev_city][next_city]
        
        cost = dist_mat[prev_city][segment[0]]
        for i in range(len(segment)-1):
            cost += dist_mat[segment[i]][segment[i+1]]
        cost += dist_mat[segment[-1]][next_city]
        return cost
    
    def _local_2opt_around_inserted(tour, inserted_set):
        for _ in range(2):
            improved = False
            for i in range(len(tour)):
                if tour[i] in inserted_set or tour[(i+1)%len(tour)] in inserted_set:
                    for j in range(i+2, len(tour) + (i if i>0 else -1)):
                        j_mod = j % len(tour)
                        if j_mod == i:
                            continue
                        if tour[j_mod] in inserted_set or tour[(j_mod+1)%len(tour)] in inserted_set:
                            a, b = tour[i], tour[(i+1)%len(tour)]
                            c, d = tour[j_mod], tour[(j_mod+1)%len(tour)]
                            
                            old_cost = dist_mat[a][b] + dist_mat[c][d]
                            new_cost = dist_mat[a][c] + dist_mat[b][d]
                            
                            if new_cost < old_cost - 1e-6:
                                if i < j_mod:
                                    tour[i+1:j_mod+1] = reversed(tour[i+1:j_mod+1])
                                else:
                                    segment = tour[i+1:] + tour[:j_mod+1]
                                    segment.reverse()
                                    tour[i+1:] = segment[:len(tour)-i-1]
                                    tour[:j_mod+1] = segment[len(tour)-i-1:]
                                improved = True
                                break
                    if improved:
                        break
            if not improved:
                break
        return tour
    
    if not removed_cities:
        return copy.deepcopy(x)
    
    shuffled_cities = copy.deepcopy(removed_cities)
    random.shuffle(shuffled_cities)
    
    result = _adaptive_regret_randomized_insertion(shuffled_cities, x)
    
    if len(removed_cities) > 0 and len(result) > 3:
        inserted_set = set(removed_cities)
        result = _local_2opt_around_inserted(result, inserted_set)
        result = _segment_optimization_vns(result, inserted_set)
    
    return result

def insert_v4(x, removed_cities, problem_data):
    if problem_data is None:
        dist_mat = None
    elif isinstance(problem_data, dict):
        dist_mat = problem_data.get('distance_matrix')
    else:
        dist_mat = problem_data
        
    if dist_mat is None:
        def insert_v1(x, removed_cities, problem_data):
            new_x = copy.deepcopy(x)
            for city in removed_cities:
                best_pos = -1
                best_increase = float('inf')
                for j in range(len(new_x) + 1):
                    if j == 0:
                        prev = new_x[-1] if new_x else city
                        nxt = new_x[0] if new_x else city
                    elif j == len(new_x):
                        prev = new_x[-1]
                        nxt = new_x[0]
                    else:
                        prev = new_x[j-1]
                        nxt = new_x[j]
                    increase = dist_mat[prev][city] + dist_mat[city][nxt] - dist_mat[prev][nxt]
                    if increase < best_increase:
                        best_increase = increase
                        best_pos = j
                new_x.insert(best_pos, city)
            return new_x
        return insert_v1(x, removed_cities, problem_data)
    
    def _calc_insertion_cost(prev_city, city, next_city):
        return dist_mat[prev_city][city] + dist_mat[city][next_city] - dist_mat[prev_city][next_city]
    
    def _compute_regret_values(cities, current_path):
        regret_data = []
        n = len(current_path)
        for city in cities:
            best_cost = float('inf')
            second_best_cost = float('inf')
            best_pos = -1
            for j in range(n + 1):
                if n == 0:
                    prev = city
                    nxt = city
                elif j == 0:
                    prev = current_path[-1]
                    nxt = current_path[0]
                elif j == n:
                    prev = current_path[-1]
                    nxt = current_path[0]
                else:
                    prev = current_path[j-1]
                    nxt = current_path[j]
                cost = _calc_insertion_cost(prev, city, nxt)
                if cost < best_cost:
                    second_best_cost = best_cost
                    best_cost = cost
                    best_pos = j
                elif cost < second_best_cost:
                    second_best_cost = cost
            regret = second_best_cost - best_cost if second_best_cost < float('inf') else 0
            normalized_regret = regret + best_cost * 0.001
            regret_data.append((city, best_cost, normalized_regret, best_pos))
        return regret_data
    
    def _adaptive_regret_temperature_insertion(candidate_cities, current_path):
        path = copy.deepcopy(current_path)
        remaining = copy.deepcopy(candidate_cities)
        n_total = len(path) + len(remaining)
        remaining_ratio = len(remaining) / n_total
        base_exploration = 0.4 * (1.0 - remaining_ratio) + 0.1
        regret_weight = 2.0 - 1.5 * remaining_ratio
        base_k = max(2, min(5, int(3 + 2 * remaining_ratio)))
        avg_dist = np.mean(dist_mat)
        temperature = avg_dist * 0.1 * remaining_ratio
        iteration = 0
        while remaining:
            iteration += 1
            regret_info = _compute_regret_values(remaining, path)
            current_exploration = base_exploration * (0.95 ** iteration)
            if len(remaining) > 1 and random.random() < current_exploration:
                k = min(base_k, len(regret_info))
                weighted_scores = []
                for city, best_cost, regret, pos in regret_info:
                    score = regret_weight * regret - best_cost
                    weighted_scores.append((score, city, best_cost, pos))
                weighted_scores.sort(reverse=True)
                top_candidates = weighted_scores[:k]
                if random.random() < 0.3 and temperature > 1e-6:
                    costs = [c[2] for c in top_candidates]
                    probs = np.exp(-np.array(costs) / temperature)
                    probs = probs / probs.sum()
                    idx = np.random.choice(len(top_candidates), p=probs)
                else:
                    if len(top_candidates) > 1:
                        scores = [c[0] for c in top_candidates]
                        max_s = max(scores)
                        min_s = min(scores)
                        if max_s > min_s:
                            normalized = [(s - min_s) / (max_s - min_s + 1e-6) for s in scores]
                            probabilities = [n / sum(normalized) for n in normalized]
                            idx = random.choices(range(len(top_candidates)), weights=probabilities)[0]
                        else:
                            idx = random.randint(0, len(top_candidates)-1)
                    else:
                        idx = 0
                selected = top_candidates[idx]
                city_to_insert = selected[1]
                insert_pos = selected[3]
            else:
                max_regret = -float('inf')
                city_to_insert = None
                insert_pos = -1
                min_best_cost = float('inf')
                for city, best_cost, regret, pos in regret_info:
                    if regret > max_regret:
                        max_regret = regret
                        city_to_insert = city
                        insert_pos = pos
                        min_best_cost = best_cost
                    elif abs(regret - max_regret) < 1e-6 and best_cost < min_best_cost:
                        city_to_insert = city
                        insert_pos = pos
                        min_best_cost = best_cost
            path.insert(insert_pos, city_to_insert)
            remaining.remove(city_to_insert)
            temperature *= 0.95
        return path
    
    def _segment_cost(segment, prev_city, next_city):
        if len(segment) == 0:
            return dist_mat[prev_city][next_city]
        cost = dist_mat[prev_city][segment[0]]
        for i in range(len(segment)-1):
            cost += dist_mat[segment[i]][segment[i+1]]
        cost += dist_mat[segment[-1]][next_city]
        return cost
    
    def _restricted_2opt_around_inserted(tour, inserted_set):
        n = len(tour)
        for _ in range(2):
            improved = False
            for i in range(n):
                if tour[i] in inserted_set or tour[(i+1)%n] in inserted_set:
                    for j in range(i+2, n + (i if i>0 else -1)):
                        j_mod = j % n
                        if j_mod == i:
                            continue
                        if tour[j_mod] in inserted_set or tour[(j_mod+1)%n] in inserted_set:
                            a, b = tour[i], tour[(i+1)%n]
                            c, d = tour[j_mod], tour[(j_mod+1)%n]
                            old_cost = dist_mat[a][b] + dist_mat[c][d]
                            new_cost = dist_mat[a][c] + dist_mat[b][d]
                            if new_cost < old_cost - 1e-6:
                                if i < j_mod:
                                    tour[i+1:j_mod+1] = reversed(tour[i+1:j_mod+1])
                                else:
                                    segment = tour[i+1:] + tour[:j_mod+1]
                                    segment.reverse()
                                    tour[i+1:] = segment[:n-i-1]
                                    tour[:j_mod+1] = segment[n-i-1:]
                                improved = True
                                break
                    if improved:
                        break
            if not improved:
                break
        return tour
    
    def _segment_optimization_vns(tour, inserted_set):
        improved = True
        iteration = 0
        n = len(tour)
        max_iterations = min(4, max(2, n // 20))
        while improved and iteration < max_iterations:
            improved = False
            base_nh = min(4, max(2, n // 15))
            for neighborhood_size in [base_nh, base_nh + 1, base_nh - 1]:
                if neighborhood_size < 2:
                    continue
                inserted_positions = [i for i in range(n) if tour[i] in inserted_set]
                for i in inserted_positions:
                    start = max(0, i - neighborhood_size)
                    end = min(n, i + neighborhood_size + 1)
                    segment = tour[start:end]
                    if len(segment) <= 3:
                        continue
                    prev_city = tour[start-1] if start > 0 else tour[-1]
                    next_city = tour[end] if end < n else tour[0]
                    current_cost = _segment_cost(segment, prev_city, next_city)
                    best_segment = segment
                    best_cost = current_cost
                    for s in range(len(segment)-1):
                        for e in range(s+2, len(segment)):
                            new_seg = segment.copy()
                            new_seg[s:e] = reversed(new_seg[s:e])
                            cost = _segment_cost(new_seg, prev_city, next_city)
                            if cost < best_cost - 1e-6:
                                best_cost = cost
                                best_segment = new_seg
                                improved = True
                    if len(segment) <= 5 and not improved:
                        for rot in range(1, len(segment)):
                            new_seg = segment[rot:] + segment[:rot]
                            cost = _segment_cost(new_seg, prev_city, next_city)
                            if cost < best_cost - 1e-6:
                                best_cost = cost
                                best_segment = new_seg
                                improved = True
                    if improved:
                        tour[start:end] = best_segment
                        break
                if improved:
                    break
            iteration += 1
        return tour
    
    def _full_tour_2opt_refinement(tour):
        n = len(tour)
        improved = True
        for _ in range(3):
            if not improved:
                break
            improved = False
            for i in range(n-1):
                for j in range(i+2, n):
                    if j == n-1 and i == 0:
                        continue
                    a, b = tour[i], tour[(i+1) % n]
                    c, d = tour[j], tour[(j+1) % n]
                    current = dist_mat[a][b] + dist_mat[c][d]
                    new = dist_mat[a][c] + dist_mat[b][d]
                    if new < current - 1e-6:
                        if j > i+1:
                            tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break
        return tour
    
    if not removed_cities:
        return copy.deepcopy(x)
    
    shuffled_cities = copy.deepcopy(removed_cities)
    random.shuffle(shuffled_cities)
    result = _adaptive_regret_temperature_insertion(shuffled_cities, x)
    
    if len(removed_cities) > 0 and len(result) > 3:
        inserted_set = set(removed_cities)
        result = _restricted_2opt_around_inserted(result, inserted_set)
        result = _segment_optimization_vns(result, inserted_set)
        result = _full_tour_2opt_refinement(result)
    
    return result

def insert_v5(x, removed_cities, problem_data):
    if problem_data is None:
        dist_mat = None
    elif isinstance(problem_data, dict):
        dist_mat = problem_data.get('distance_matrix')
    else:
        dist_mat = problem_data
        
    if dist_mat is None:
        def insert_v1(x, removed_cities, problem_data):
            new_x = copy.deepcopy(x)
            for city in removed_cities:
                best_pos = -1
                best_increase = float('inf')
                for j in range(len(new_x) + 1):
                    if j == 0:
                        prev = new_x[-1] if new_x else city
                        nxt = new_x[0] if new_x else city
                    elif j == len(new_x):
                        prev = new_x[-1]
                        nxt = new_x[0]
                    else:
                        prev = new_x[j-1]
                        nxt = new_x[j]
                    increase = dist_mat[prev][city] + dist_mat[city][nxt] - dist_mat[prev][nxt]
                    if increase < best_increase:
                        best_increase = increase
                        best_pos = j
                new_x.insert(best_pos, city)
            return new_x
        return insert_v1(x, removed_cities, problem_data)
    
    def _calc_insertion_cost(prev_city, city, next_city):
        return dist_mat[prev_city][city] + dist_mat[city][next_city] - dist_mat[prev_city][next_city]
    
    def _tour_distance(path):
        if len(path) <= 1:
            return 0
        d = 0
        for i in range(len(path) - 1):
            d += dist_mat[path[i]][path[i + 1]]
        d += dist_mat[path[-1]][path[0]]
        return d
    
    def _adaptive_parameter_tuning(remaining_count, total_to_insert):
        """Dynamically adjust parameters based on insertion progress"""
        progress = 1.0 - (remaining_count / max(total_to_insert, 1))
        
        # Randomization: high initially, decreases for fine-tuning
        base_rand = 0.25
        rand_factor = base_rand * (1.0 - 0.7 * progress)
        
        # Greediness: increases as we progress
        greediness = 0.3 + 0.6 * progress
        
        # Temperature: starts high, cools quickly
        temp_base = np.mean(dist_mat) * 0.08
        temperature = temp_base * (1.0 - 0.8 * progress)
        
        # Candidate pool size: larger initially, smaller later
        candidate_pool = max(2, min(5, int(3 + 2 * (1.0 - progress))))
        
        return rand_factor, greediness, temperature, candidate_pool
    
    def _regret_insertion_with_adaptive_randomization(candidate_cities, current_path):
        """Regret-2 insertion with adaptive randomization and greedy thresholds"""
        path = copy.deepcopy(current_path)
        remaining = copy.deepcopy(candidate_cities)
        
        while remaining:
            # Update adaptive parameters
            rand_factor, greediness, temperature, candidate_pool = _adaptive_parameter_tuning(
                len(remaining), len(candidate_cities)
            )
            
            # Select candidate subset
            if len(remaining) > candidate_pool:
                if random.random() < rand_factor:
                    candidates = random.sample(remaining, candidate_pool)
                else:
                    # Select candidates with smallest nearest neighbor distances
                    nn_distances = []
                    for city in remaining:
                        min_dist = min([dist_mat[city][p] for p in path] if path else [0])
                        nn_distances.append((min_dist, city))
                    nn_distances.sort()
                    candidates = [city for _, city in nn_distances[:candidate_pool]]
            else:
                candidates = list(remaining)
            
            best_city = None
            best_pos = -1
            best_regret = -float('inf')
            city_data = {}
            
            # Calculate regret values for candidates
            for city in candidates:
                insertion_costs = []
                
                for j in range(len(path) + 1):
                    if j == 0:
                        prev = path[-1] if path else city
                        nxt = path[0] if path else city
                    elif j == len(path):
                        prev = path[-1]
                        nxt = path[0]
                    else:
                        prev = path[j-1]
                        nxt = path[j]
                    
                    cost = _calc_insertion_cost(prev, city, nxt)
                    insertion_costs.append((cost, j))
                
                insertion_costs.sort()
                best_cost, best_j = insertion_costs[0]
                
                # Regret-2: difference between best and second best
                regret = 0
                if len(insertion_costs) > 1:
                    regret = insertion_costs[1][0] - best_cost
                
                city_data[city] = {
                    'best_cost': best_cost,
                    'best_pos': best_j,
                    'regret': regret,
                    'all_costs': insertion_costs
                }
                
                # Apply greedy threshold: sometimes choose based on regret, sometimes on absolute cost
                score = 0
                if random.random() < greediness:
                    score = -best_cost  # Greedy on cost
                else:
                    score = regret  # Regret-based
                
                if score > best_regret:
                    best_regret = score
                    best_city = city
                    best_pos = best_j
            
            # Apply probabilistic acceptance of alternative positions
            if best_city is not None and len(city_data[best_city]['all_costs']) > 1:
                alt_costs = city_data[best_city]['all_costs'][:3]  # Top 3 alternatives
                if len(alt_costs) > 1 and temperature > 1e-6:
                    best_alt_cost, best_alt_pos = alt_costs[0]
                    for alt_cost, alt_pos in alt_costs[1:]:
                        cost_diff = alt_cost - best_alt_cost
                        if cost_diff > 0:
                            accept_prob = math.exp(-cost_diff / temperature)
                            if random.random() < accept_prob * rand_factor:
                                best_pos = alt_pos
                                break
            
            # Insert the selected city
            path.insert(best_pos, best_city)
            remaining.remove(best_city)
    
        return path
    
    def _focused_local_improvement(path, inserted_set):
        """Apply limited local search around inserted cities"""
        if len(inserted_set) == 0 or len(path) < 4:
            return path
        
        improved = True
        iterations = 0
        max_iterations = min(3, len(inserted_set))
        
        while improved and iterations < max_iterations:
            improved = False
            
            # Create list of positions containing inserted cities
            inserted_positions = [i for i, city in enumerate(path) if city in inserted_set]
            
            # Try 2-opt swaps involving inserted cities
            for idx, i in enumerate(inserted_positions):
                for j in inserted_positions[idx+1:]:
                    if abs(i - j) < 2:
                        continue
                    
                    # Ensure we have valid indices
                    i1, i2 = i, (i + 1) % len(path)
                    j1, j2 = j, (j + 1) % len(path)
                    
                    a, b = path[i1], path[i2]
                    c, d = path[j1], path[j2]
                    
                    old_cost = dist_mat[a][b] + dist_mat[c][d]
                    new_cost = dist_mat[a][c] + dist_mat[b][d]
                    
                    if new_cost < old_cost - 1e-6:
                        # Perform 2-opt swap
                        if i1 < j1:
                            path[i1+1:j1+1] = reversed(path[i1+1:j1+1])
                        else:
                            segment = path[i1+1:] + path[:j1+1]
                            segment.reverse()
                            path[i1+1:] = segment[:len(path)-i1-1]
                            path[:j1+1] = segment[len(path)-i1-1:]
                        
                        improved = True
                        break
                
                if improved:
                    break
            
            iterations += 1
        
        return path
    
    # Shuffle removed cities to avoid order bias
    shuffled_cities = copy.deepcopy(removed_cities)
    random.shuffle(shuffled_cities)
    
    # Apply adaptive regret insertion
    result = _regret_insertion_with_adaptive_randomization(shuffled_cities, x)
    
    # Apply focused local improvement
    inserted_set = set(removed_cities)
    result = _focused_local_improvement(result, inserted_set)
    
    return result
