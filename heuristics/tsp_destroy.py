# Best Solution: 5.71
import copy
import numpy as np
import random
import math

def destroy_v1(x, destroy_cnt, problem_data=None):
    new_x = copy.deepcopy(x)
    removed_cities = []
    
    if len(x) <= destroy_cnt:
        return list(range(len(x))), []
    
    def calculate_segment_cost(tour, start_idx, length):
        cost = 0
        for i in range(length - 1):
            idx1 = (start_idx + i) % len(tour)
            idx2 = (start_idx + i + 1) % len(tour)
            cost += problem_data['distance_matrix'][tour[idx1], tour[idx2]]
        return cost
    
    def find_high_cost_segments(tour, num_candidates=15):
        segment_length = max(3, len(tour) // 10)
        segment_costs = []
        
        for i in range(len(tour)):
            cost = calculate_segment_cost(tour, i, segment_length)
            segment_costs.append((cost, i))
        
        segment_costs.sort(reverse=True, key=lambda x: x[0])
        return [sc[1] for sc in segment_costs[:num_candidates]]
    
    def adaptive_destroy_size(base_cnt, tour_length):
        if tour_length < 50:
            return min(base_cnt, tour_length // 2)
        elif tour_length < 150:
            return min(base_cnt + 6, tour_length // 3)
        else:
            return min(base_cnt + 12, tour_length // 3)
    
    def calculate_city_density(city_idx, coordinates, k_neighbors=10):
        if coordinates is None:
            return 0
        point = coordinates[city_idx]
        distances = np.linalg.norm(coordinates - point, axis=1)
        distances[city_idx] = np.inf
        nearest_distances = np.partition(distances, k_neighbors)[:k_neighbors]
        density = 1.0 / (np.mean(nearest_distances) + 1e-10)
        return density
    
    def calculate_removal_score(city_idx, tour_idx, coordinates, distance_matrix, current_tour):
        score = 0
        
        if coordinates is not None:
            density = calculate_city_density(city_idx, coordinates)
            score += density * 0.25
        
        if distance_matrix is not None:
            prev_city = current_tour[(tour_idx - 1) % len(current_tour)]
            next_city = current_tour[(tour_idx + 1) % len(current_tour)]
            
            current_cost = distance_matrix[prev_city, city_idx] + distance_matrix[city_idx, next_city]
            direct_cost = distance_matrix[prev_city, next_city]
            detour_cost = current_cost - direct_cost
            score += detour_cost * 0.75
        
        return score
    
    actual_destroy_cnt = adaptive_destroy_size(destroy_cnt, len(x))
    
    if problem_data is not None and 'distance_matrix' in problem_data:
        distance_matrix = problem_data['distance_matrix']
        coordinates = problem_data.get('coordinates')
        
        removal_strategy = random.random()
        tour = x
        
        if removal_strategy < 0.65:
            high_cost_starts = find_high_cost_segments(tour)
            start_idx = random.choice(high_cost_starts)
            
            removal_indices = []
            for i in range(actual_destroy_cnt):
                idx = (start_idx + i) % len(tour)
                removal_indices.append(idx)
            
            removal_indices.sort(reverse=True)
            for idx in removal_indices:
                removed_cities.append(new_x[idx])
                new_x.pop(idx)
                
        elif removal_strategy < 0.90:
            city_scores = []
            for i, city in enumerate(tour):
                score = calculate_removal_score(city, i, coordinates, distance_matrix, tour)
                city_scores.append((score, i, city))
            
            city_scores.sort(reverse=True, key=lambda x: x[0])
            top_candidates = [cs[1] for cs in city_scores[:actual_destroy_cnt * 3]]
            
            if len(top_candidates) > 0:
                selected_indices = random.sample(top_candidates, min(actual_destroy_cnt, len(top_candidates)))
                selected_indices.sort(reverse=True)
                
                for idx in selected_indices:
                    removed_cities.append(new_x[idx])
                    new_x.pop(idx)
                
                remaining = actual_destroy_cnt - len(selected_indices)
                if remaining > 0:
                    available_indices = [i for i in range(len(new_x))]
                    if len(available_indices) > 0:
                        random_indices = random.sample(available_indices, min(remaining, len(available_indices)))
                        random_indices.sort(reverse=True)
                        for idx in random_indices:
                            removed_cities.append(new_x[idx])
                            new_x.pop(idx)
            else:
                available_indices = list(range(len(new_x)))
                if len(available_indices) > 0:
                    random_indices = random.sample(available_indices, min(actual_destroy_cnt, len(available_indices)))
                    random_indices.sort(reverse=True)
                    for idx in random_indices:
                        removed_cities.append(new_x[idx])
                        new_x.pop(idx)
        else:
            start_idx = random.randint(0, len(tour) - 1)
            removal_indices = []
            
            for i in range(actual_destroy_cnt):
                idx = (start_idx + i) % len(tour)
                removal_indices.append(idx)
            
            removal_indices.sort(reverse=True)
            for idx in removal_indices:
                removed_cities.append(new_x[idx])
                new_x.pop(idx)
    else:
        start_idx = random.randint(0, len(new_x) - actual_destroy_cnt)
        for i in range(start_idx + actual_destroy_cnt - 1, start_idx - 1, -1):
            removed_cities.append(new_x[i])
            new_x.pop(i)
    
    return removed_cities, new_x

def destroy_v2(x, destroy_cnt, problem_data=None):
    new_x = copy.deepcopy(x)
    removed_cities = []
    
    if len(x) <= destroy_cnt:
        return list(range(len(x))), []
    
    def adaptive_destroy_size(base_cnt, tour_length):
        if tour_length < 50:
            return min(base_cnt, tour_length // 2)
        elif tour_length < 150:
            return min(base_cnt + 8, tour_length // 3)
        else:
            return min(base_cnt + 15, tour_length // 3)
    
    def calculate_segment_cost(tour, start_idx, length):
        cost = 0
        for i in range(length - 1):
            idx1 = (start_idx + i) % len(tour)
            idx2 = (start_idx + i + 1) % len(tour)
            cost += problem_data['distance_matrix'][tour[idx1], tour[idx2]]
        return cost
    
    def find_high_cost_segments(tour, num_candidates=20):
        segment_length = max(3, len(tour) // 8)
        segment_costs = []
        for i in range(len(tour)):
            cost = calculate_segment_cost(tour, i, segment_length)
            segment_costs.append((cost, i))
        segment_costs.sort(reverse=True, key=lambda x: x[0])
        return [sc[1] for sc in segment_costs[:num_candidates]]
    
    def calculate_city_density(city_idx, coordinates, k_neighbors=10):
        point = coordinates[city_idx]
        distances = np.linalg.norm(coordinates - point, axis=1)
        distances[city_idx] = np.inf
        nearest_distances = np.partition(distances, k_neighbors)[:k_neighbors]
        density = 1.0 / (np.mean(nearest_distances) + 1e-10)
        return density
    
    def calculate_spatial_dispersion(city_idx, coordinates, k_neighbors=8):
        point = coordinates[city_idx]
        distances = np.linalg.norm(coordinates - point, axis=1)
        distances[city_idx] = np.inf
        nearest_distances = np.partition(distances, k_neighbors)[:k_neighbors]
        return np.std(nearest_distances) / (np.mean(nearest_distances) + 1e-10)
    
    def calculate_removal_score(city_idx, tour_idx, coordinates, distance_matrix, current_tour):
        score = 0
        if coordinates is not None:
            density = calculate_city_density(city_idx, coordinates)
            score += density * 0.2
            dispersion = calculate_spatial_dispersion(city_idx, coordinates)
            score += dispersion * 0.15
        if distance_matrix is not None:
            prev_city = current_tour[(tour_idx - 1) % len(current_tour)]
            next_city = current_tour[(tour_idx + 1) % len(current_tour)]
            current_cost = distance_matrix[prev_city, city_idx] + distance_matrix[city_idx, next_city]
            direct_cost = distance_matrix[prev_city, next_city]
            detour_cost = current_cost - direct_cost
            score += detour_cost * 0.65
        return score
    
    def calculate_tour_curvature(tour, coordinates):
        if coordinates is None or len(tour) < 3:
            return np.zeros(len(tour))
        curvatures = np.zeros(len(tour))
        for i in range(len(tour)):
            a = coordinates[tour[(i-1) % len(tour)]]
            b = coordinates[tour[i]]
            c = coordinates[tour[(i+1) % len(tour)]]
            ba = a - b
            bc = c - b
            dot = np.dot(ba, bc)
            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)
            if norm_ba > 0 and norm_bc > 0:
                angle = math.acos(max(-1, min(1, dot/(norm_ba*norm_bc))))
                curvatures[i] = math.pi - angle
        return curvatures
    
    def identify_bottleneck_regions(tour, distance_matrix, window_size=6):
        scores = []
        n = len(tour)
        for i in range(n):
            segment_cost = 0
            for j in range(window_size):
                idx1 = (i + j) % n
                idx2 = (i + j + 1) % n
                segment_cost += distance_matrix[tour[idx1], tour[idx2]]
            avg_cost = segment_cost / window_size
            max_edge = max(distance_matrix[tour[(i+j)%n], tour[(i+j+1)%n]] for j in range(window_size))
            scores.append(avg_cost * max_edge)
        threshold = np.percentile(scores, 80)
        return [i for i, score in enumerate(scores) if score > threshold]
    
    def find_cluster_centers(coordinates, k=6):
        from sklearn.cluster import KMeans
        k = min(k, len(coordinates))
        if k <= 1:
            return []
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(coordinates)
        centers = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(coordinates - center, axis=1)
            centers.append(np.argmin(distances))
        return centers
    
    actual_destroy_cnt = adaptive_destroy_size(destroy_cnt, len(x))
    
    if problem_data is not None and 'distance_matrix' in problem_data:
        distance_matrix = problem_data['distance_matrix']
        coordinates = problem_data.get('coordinates')
        tour = x
        
        strategy_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        strategy = random.choices(range(5), weights=strategy_weights)[0]
        
        if strategy == 0:  # High-cost segment removal (enhanced)
            high_cost_starts = find_high_cost_segments(tour)
            if high_cost_starts:
                start_idx = random.choice(high_cost_starts)
                removal_indices = [(start_idx + i) % len(tour) for i in range(actual_destroy_cnt)]
            else:
                removal_indices = random.sample(range(len(tour)), min(actual_destroy_cnt, len(tour)))
        
        elif strategy == 1:  # Curvature-dispersion hybrid
            if coordinates is not None:
                curvatures = calculate_tour_curvature(tour, coordinates)
                high_curv_indices = np.argsort(curvatures)[-actual_destroy_cnt*3:]
                if len(high_curv_indices) > 0:
                    dispersion_scores = []
                    for idx in high_curv_indices:
                        city = tour[idx]
                        dispersion = calculate_spatial_dispersion(city, coordinates)
                        impact = calculate_removal_score(city, idx, coordinates, distance_matrix, tour)
                        combined = curvatures[idx] * 0.6 + dispersion * 0.2 + impact * 0.2
                        dispersion_scores.append((combined, idx))
                    dispersion_scores.sort(reverse=True, key=lambda x: x[0])
                    top_candidates = [idx for _, idx in dispersion_scores[:actual_destroy_cnt*2]]
                    removal_indices = random.sample(top_candidates, min(actual_destroy_cnt, len(top_candidates)))
                else:
                    removal_indices = random.sample(range(len(tour)), min(actual_destroy_cnt, len(tour)))
            else:
                removal_indices = random.sample(range(len(tour)), min(actual_destroy_cnt, len(tour)))
        
        elif strategy == 2:  # Bottleneck-aware scoring
            bottleneck_indices = identify_bottleneck_regions(tour, distance_matrix)
            if bottleneck_indices:
                start_idx = random.choice(bottleneck_indices)
                base_indices = [(start_idx + i) % len(tour) for i in range(actual_destroy_cnt // 2)]
                remaining = actual_destroy_cnt - len(base_indices)
                if remaining > 0:
                    city_scores = []
                    for i, city in enumerate(tour):
                        if i not in base_indices:
                            score = calculate_removal_score(city, i, coordinates, distance_matrix, tour)
                            city_scores.append((score, i))
                    city_scores.sort(reverse=True, key=lambda x: x[0])
                    top_candidates = [idx for _, idx in city_scores[:remaining*3]]
                    if len(top_candidates) > 0:
                        additional = random.sample(top_candidates, min(remaining, len(top_candidates)))
                        base_indices.extend(additional)
                removal_indices = base_indices
            else:
                removal_indices = random.sample(range(len(tour)), min(actual_destroy_cnt, len(tour)))
        
        elif strategy == 3:  # Cluster-center focused removal
            if coordinates is not None:
                cluster_centers = find_cluster_centers(coordinates)
                center_indices = [i for i, city in enumerate(tour) if city in cluster_centers]
                if center_indices:
                    start_idx = random.choice(center_indices)
                    removal_indices = [(start_idx + i) % len(tour) for i in range(actual_destroy_cnt)]
                else:
                    density_scores = []
                    for i, city in enumerate(tour):
                        density = calculate_city_density(city, coordinates)
                        impact = calculate_removal_score(city, i, coordinates, distance_matrix, tour)
                        combined = density * 0.4 + impact * 0.6
                        density_scores.append((combined, i))
                    density_scores.sort(reverse=True, key=lambda x: x[0])
                    removal_indices = [idx for _, idx in density_scores[:actual_destroy_cnt]]
            else:
                removal_indices = random.sample(range(len(tour)), min(actual_destroy_cnt, len(tour)))
        
        else:  # Adaptive hybrid random
            removal_indices = []
            remaining = actual_destroy_cnt
            if coordinates is not None:
                curvatures = calculate_tour_curvature(tour, coordinates)
                high_curv = np.argsort(curvatures)[-actual_destroy_cnt//3:]
                if len(high_curv) > 0:
                    selected = random.sample(list(high_curv), min(actual_destroy_cnt//3, len(high_curv)))
                    removal_indices.extend(selected)
                    remaining -= len(selected)
            if remaining > 0:
                city_scores = []
                for i, city in enumerate(tour):
                    if i not in removal_indices:
                        score = calculate_removal_score(city, i, coordinates, distance_matrix, tour)
                        city_scores.append((score, i))
                city_scores.sort(reverse=True, key=lambda x: x[0])
                top_candidates = [idx for _, idx in city_scores[:remaining*4]]
                if len(top_candidates) > 0:
                    additional = random.sample(top_candidates, min(remaining, len(top_candidates)))
                    removal_indices.extend(additional)
                    remaining -= len(additional)
            if remaining > 0:
                available = [i for i in range(len(tour)) if i not in removal_indices]
                if len(available) > 0:
                    extra = random.sample(available, min(remaining, len(available)))
                    removal_indices.extend(extra)
        
        removal_indices = list(set(removal_indices))
        if len(removal_indices) > actual_destroy_cnt:
            removal_indices = random.sample(removal_indices, actual_destroy_cnt)
        removal_indices.sort(reverse=True)
        
        for idx in removal_indices:
            removed_cities.append(new_x[idx])
            new_x.pop(idx)
    
    else:
        start_idx = random.randint(0, len(new_x) - actual_destroy_cnt)
        for i in range(start_idx + actual_destroy_cnt - 1, start_idx - 1, -1):
            removed_cities.append(new_x[i])
            new_x.pop(i)
    
    return removed_cities, new_x

def destroy_v3(x, destroy_cnt, problem_data=None):
    new_x = copy.deepcopy(x)
    removed_cities = []
    
    if len(x) <= destroy_cnt:
        return list(range(len(x))), []
    
    def adaptive_destroy_size(base_cnt, tour_length):
        if tour_length < 50:
            return min(base_cnt, tour_length // 2)
        elif tour_length < 150:
            return min(base_cnt + 8, tour_length // 3)
        else:
            return min(base_cnt + 15, tour_length // 4)
    
    def calculate_segment_cost(tour, start_idx, length, distance_matrix):
        cost = 0
        for i in range(length - 1):
            idx1 = (start_idx + i) % len(tour)
            idx2 = (start_idx + i + 1) % len(tour)
            cost += distance_matrix[tour[idx1], tour[idx2]]
        return cost
    
    def find_bottleneck_segments(tour, distance_matrix, num_candidates=12):
        segment_length = max(3, len(tour) // 8)
        segment_scores = []
        
        for i in range(len(tour)):
            cost = calculate_segment_cost(tour, i, segment_length, distance_matrix)
            max_edge = max(distance_matrix[tour[(i+j)%len(tour)], tour[(i+j+1)%len(tour)]] for j in range(segment_length))
            score = cost * max_edge / segment_length
            segment_scores.append((score, i))
        
        segment_scores.sort(reverse=True, key=lambda x: x[0])
        return [ss[1] for ss in segment_scores[:num_candidates]]
    
    def calculate_tour_curvature(tour, coordinates):
        if coordinates is None or len(tour) < 3:
            return np.zeros(len(tour))
        
        curvatures = np.zeros(len(tour))
        for i in range(len(tour)):
            a = coordinates[tour[(i-1) % len(tour)]]
            b = coordinates[tour[i]]
            c = coordinates[tour[(i+1) % len(tour)]]
            
            ba = a - b
            bc = c - b
            dot = np.dot(ba, bc)
            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)
            
            if norm_ba > 0 and norm_bc > 0:
                angle = math.acos(max(-1, min(1, dot/(norm_ba*norm_bc))))
                curvatures[i] = math.pi - angle
        return curvatures
    
    def calculate_spatial_density(city_idx, coordinates, k_neighbors=10):
        if coordinates is None:
            return 0
        point = coordinates[city_idx]
        distances = np.linalg.norm(coordinates - point, axis=1)
        distances[city_idx] = np.inf
        nearest_distances = np.partition(distances, k_neighbors)[:k_neighbors]
        density = 1.0 / (np.mean(nearest_distances) + 1e-10)
        return density
    
    def calculate_removal_impact(city_idx, tour_idx, distance_matrix, tour):
        prev_city = tour[(tour_idx - 1) % len(tour)]
        next_city = tour[(tour_idx + 1) % len(tour)]
        current_cost = distance_matrix[prev_city, city_idx] + distance_matrix[city_idx, next_city]
        direct_cost = distance_matrix[prev_city, next_city]
        return current_cost - direct_cost
    
    def compute_composite_score(city_idx, tour_idx, tour, distance_matrix, coordinates, curvature_values):
        score = 0
        
        impact = calculate_removal_impact(city_idx, tour_idx, distance_matrix, tour)
        score += impact * 0.4
        
        if coordinates is not None:
            density = calculate_spatial_density(city_idx, coordinates)
            score += density * 0.2
            
            if curvature_values is not None:
                score += curvature_values[tour_idx] * 0.4
        else:
            score += random.random() * 0.1
        
        return score
    
    actual_destroy_cnt = adaptive_destroy_size(destroy_cnt, len(x))
    
    if problem_data is not None and 'distance_matrix' in problem_data:
        distance_matrix = problem_data['distance_matrix']
        coordinates = problem_data.get('coordinates')
        tour = x
        
        strategy_weights = [0.40, 0.30, 0.20, 0.10]
        strategy = random.choices(range(4), weights=strategy_weights)[0]
        
        if strategy == 0:  # Bottleneck segment removal
            bottleneck_starts = find_bottleneck_segments(tour, distance_matrix)
            if bottleneck_starts:
                start_idx = random.choice(bottleneck_starts)
                removal_indices = [(start_idx + i) % len(tour) for i in range(actual_destroy_cnt)]
            else:
                removal_indices = random.sample(range(len(tour)), min(actual_destroy_cnt, len(tour)))
                
        elif strategy == 1:  # Composite score-based removal
            curvature_values = None
            if coordinates is not None:
                curvature_values = calculate_tour_curvature(tour, coordinates)
            
            city_scores = []
            for i, city in enumerate(tour):
                score = compute_composite_score(city, i, tour, distance_matrix, coordinates, curvature_values)
                city_scores.append((score, i))
            
            city_scores.sort(reverse=True, key=lambda x: x[0])
            top_candidates = [cs[1] for cs in city_scores[:actual_destroy_cnt * 3]]
            
            if len(top_candidates) > 0:
                removal_indices = random.sample(top_candidates, min(actual_destroy_cnt, len(top_candidates)))
            else:
                removal_indices = random.sample(range(len(tour)), min(actual_destroy_cnt, len(tour)))
                
        elif strategy == 2:  # High-curvature focused removal
            if coordinates is not None:
                curvatures = calculate_tour_curvature(tour, coordinates)
                high_curvature_indices = np.argsort(curvatures)[-actual_destroy_cnt*2:]
                removal_indices = random.sample(list(high_curvature_indices), 
                                               min(actual_destroy_cnt, len(high_curvature_indices)))
            else:
                removal_indices = random.sample(range(len(tour)), min(actual_destroy_cnt, len(tour)))
                
        else:  # Hybrid: bottleneck start + random expansion
            removal_indices = []
            if len(tour) > 0:
                start_idx = random.randint(0, len(tour) - 1)
                segment_size = min(actual_destroy_cnt // 2, len(tour))
                for i in range(segment_size):
                    removal_indices.append((start_idx + i) % len(tour))
                
                remaining = actual_destroy_cnt - len(removal_indices)
                if remaining > 0:
                    available = [i for i in range(len(tour)) if i not in removal_indices]
                    if len(available) > 0:
                        additional = random.sample(available, min(remaining, len(available)))
                        removal_indices.extend(additional)
        
        removal_indices = list(set(removal_indices))
        removal_indices.sort(reverse=True)
        
        for idx in removal_indices[:actual_destroy_cnt]:
            if idx < len(new_x):
                removed_cities.append(new_x[idx])
                new_x.pop(idx)
                
    else:
        start_idx = random.randint(0, len(new_x) - actual_destroy_cnt)
        for i in range(start_idx + actual_destroy_cnt - 1, start_idx - 1, -1):
            removed_cities.append(new_x[i])
            new_x.pop(i)
    
    return removed_cities, new_x

def destroy_v4(x, destroy_cnt, problem_data=None):
    new_x = copy.deepcopy(x)
    removed_cities = []
    
    if len(x) <= destroy_cnt:
        return list(range(len(x))), []
    
    def adaptive_destroy_size(base_cnt, tour_length):
        if tour_length < 30:
            return min(base_cnt, tour_length // 2)
        elif tour_length < 100:
            return min(base_cnt + 5, tour_length // 3)
        elif tour_length < 300:
            return min(base_cnt + 10, tour_length // 4)
        else:
            return min(base_cnt + 15, tour_length // 5)
    
    def calculate_edge_weights(tour, distance_matrix):
        n = len(tour)
        weights = np.zeros(n)
        for i in range(n):
            weights[i] = distance_matrix[tour[i], tour[(i+1)%n]]
        return weights
    
    def find_promising_removal_regions(tour, distance_matrix, num_regions=8):
        n = len(tour)
        edge_weights = calculate_edge_weights(tour, distance_matrix)
        
        # Use sliding window to find high-cost segments
        window_size = max(3, n // 10)
        segment_scores = []
        
        for i in range(n):
            segment_indices = [(i + j) % n for j in range(window_size)]
            segment_cost = sum(edge_weights[idx % n] for idx in segment_indices[:-1])
            
            # Consider both average and maximum edge weight
            avg_cost = segment_cost / (window_size - 1)
            max_cost = max(edge_weights[idx % n] for idx in segment_indices[:-1])
            
            # Score favors segments with both high average and high maximum
            score = avg_cost * max_cost * (window_size - 1)
            segment_scores.append((score, i))
        
        segment_scores.sort(reverse=True, key=lambda x: x[0])
        return [ss[1] for ss in segment_scores[:num_regions]]
    
    def calculate_local_improvement_potential(city_idx, tour_idx, tour, distance_matrix):
        n = len(tour)
        prev_city = tour[(tour_idx - 1) % n]
        next_city = tour[(tour_idx + 1) % n]
        city = tour[tour_idx]
        
        current_cost = distance_matrix[prev_city, city] + distance_matrix[city, next_city]
        direct_cost = distance_matrix[prev_city, next_city]
        
        # Also consider alternative connections
        alt_prev = tour[(tour_idx - 2) % n]
        alt_next = tour[(tour_idx + 2) % n]
        
        alt_cost1 = distance_matrix[alt_prev, city] + distance_matrix[city, next_city]
        alt_cost2 = distance_matrix[prev_city, city] + distance_matrix[city, alt_next]
        
        best_alternative = min(alt_cost1, alt_cost2, direct_cost)
        return current_cost - best_alternative
    
    def compute_spatial_clustering(tour, coordinates, k_neighbors=8):
        n = len(tour)
        clustering_scores = np.zeros(n)
        
        for i in range(n):
            city = tour[i]
            point = coordinates[city]
            
            # Calculate distances to all other cities in tour
            tour_coords = coordinates[tour]
            distances = np.linalg.norm(tour_coords - point, axis=1)
            distances[i] = np.inf
            
            # Find k nearest neighbors in the current tour
            nearest_indices = np.argpartition(distances, k_neighbors)[:k_neighbors]
            avg_distance = np.mean(distances[nearest_indices])
            
            # Score is inverse of average distance (higher = more clustered)
            clustering_scores[i] = 1.0 / (avg_distance + 1e-10)
        
        return clustering_scores
    
    def adaptive_removal_strategy(tour, destroy_cnt, distance_matrix, coordinates):
        n = len(tour)
        removal_indices = []
        
        # Phase 1: Identify and remove from worst segments
        bad_segments = find_promising_removal_regions(tour, distance_matrix)
        segment_removals = min(destroy_cnt // 2, len(bad_segments) * 3)
        
        for start_idx in bad_segments[:segment_removals//3]:
            segment_size = min(3, destroy_cnt - len(removal_indices))
            for offset in range(segment_size):
                idx = (start_idx + offset) % n
                if idx not in removal_indices and len(removal_indices) < destroy_cnt:
                    removal_indices.append(idx)
        
        # Phase 2: Score-based removal for remaining slots
        remaining = destroy_cnt - len(removal_indices)
        if remaining > 0:
            scores = []
            for i in range(n):
                if i not in removal_indices:
                    # Improvement potential score
                    imp_score = calculate_local_improvement_potential(tour[i], i, tour, distance_matrix)
                    
                    # Spatial clustering score (if coordinates available)
                    cluster_score = 0
                    if coordinates is not None:
                        clustering_scores = compute_spatial_clustering(tour, coordinates)
                        cluster_score = clustering_scores[i]
                    
                    # Edge weight consideration
                    prev_edge = distance_matrix[tour[(i-1)%n], tour[i]]
                    next_edge = distance_matrix[tour[i], tour[(i+1)%n]]
                    edge_score = (prev_edge + next_edge) / 2
                    
                    # Composite score
                    total_score = imp_score * 0.5 + cluster_score * 0.3 + edge_score * 0.2
                    scores.append((total_score, i))
            
            scores.sort(reverse=True, key=lambda x: x[0])
            top_candidates = [idx for _, idx in scores[:remaining * 2]]
            
            if top_candidates:
                selected = random.sample(top_candidates, min(remaining, len(top_candidates)))
                removal_indices.extend(selected)
        
        # Phase 3: Fill any remaining slots randomly
        if len(removal_indices) < destroy_cnt:
            available = [i for i in range(n) if i not in removal_indices]
            if available:
                additional = random.sample(available, min(destroy_cnt - len(removal_indices), len(available)))
                removal_indices.extend(additional)
        
        return removal_indices[:destroy_cnt]
    
    actual_destroy_cnt = adaptive_destroy_size(destroy_cnt, len(x))
    
    if problem_data is not None and 'distance_matrix' in problem_data:
        distance_matrix = problem_data['distance_matrix']
        coordinates = problem_data.get('coordinates')
        tour = x
        
        # Use adaptive strategy based on problem size
        if len(tour) < 50:
            # For small problems, use more aggressive removal
            removal_indices = adaptive_removal_strategy(tour, actual_destroy_cnt, distance_matrix, coordinates)
        else:
            # For larger problems, use probabilistic strategy selection
            strategies = [
                lambda: adaptive_removal_strategy(tour, actual_destroy_cnt, distance_matrix, coordinates),
                lambda: random.sample(range(len(tour)), min(actual_destroy_cnt, len(tour))),
                lambda: list(np.random.choice(len(tour), min(actual_destroy_cnt, len(tour)), replace=False))
            ]
            weights = [0.6, 0.25, 0.15]
            strategy_idx = random.choices(range(len(strategies)), weights=weights)[0]
            removal_indices = strategies[strategy_idx]()
        
        # Ensure unique indices and sort for safe removal
        removal_indices = list(set(removal_indices))
        if len(removal_indices) > actual_destroy_cnt:
            removal_indices = random.sample(removal_indices, actual_destroy_cnt)
        removal_indices.sort(reverse=True)
        
        # Perform removal
        for idx in removal_indices:
            if 0 <= idx < len(new_x):
                removed_cities.append(new_x[idx])
                new_x.pop(idx)
    else:
        # Fallback for missing problem data
        start_idx = random.randint(0, len(new_x) - actual_destroy_cnt)
        for i in range(start_idx + actual_destroy_cnt - 1, start_idx - 1, -1):
            removed_cities.append(new_x[i])
            new_x.pop(i)
    
    return removed_cities, new_x

def destroy_v5(x, destroy_cnt, problem_data=None):
    new_x = copy.deepcopy(x)
    removed_cities = []
    
    if len(x) <= destroy_cnt:
        return list(range(len(x))), []
    
    def adaptive_destroy_size(base_cnt, tour_length):
        if tour_length < 30:
            return min(base_cnt, tour_length // 2)
        elif tour_length < 100:
            return min(base_cnt + 5, tour_length // 3)
        elif tour_length < 300:
            return min(base_cnt + 10, tour_length // 4)
        else:
            return min(base_cnt + 15, tour_length // 5)
    
    def calculate_edge_weights(tour, distance_matrix):
        n = len(tour)
        weights = np.zeros(n)
        for i in range(n):
            weights[i] = distance_matrix[tour[i], tour[(i+1)%n]]
        return weights
    
    def find_promising_removal_regions(tour, distance_matrix, num_regions=8):
        n = len(tour)
        edge_weights = calculate_edge_weights(tour, distance_matrix)
        
        # Use sliding window to find high-cost segments
        window_size = max(3, n // 10)
        segment_scores = []
        
        for i in range(n):
            segment_indices = [(i + j) % n for j in range(window_size)]
            segment_cost = sum(edge_weights[idx % n] for idx in segment_indices[:-1])
            
            # Consider both average and maximum edge weight
            avg_cost = segment_cost / (window_size - 1)
            max_cost = max(edge_weights[idx % n] for idx in segment_indices[:-1])
            
            # Score favors segments with both high average and high maximum
            score = avg_cost * max_cost * (window_size - 1)
            segment_scores.append((score, i))
        
        segment_scores.sort(reverse=True, key=lambda x: x[0])
        return [ss[1] for ss in segment_scores[:num_regions]]
    
    def calculate_local_improvement_potential(city_idx, tour_idx, tour, distance_matrix):
        n = len(tour)
        prev_city = tour[(tour_idx - 1) % n]
        next_city = tour[(tour_idx + 1) % n]
        city = tour[tour_idx]
        
        current_cost = distance_matrix[prev_city, city] + distance_matrix[city, next_city]
        direct_cost = distance_matrix[prev_city, next_city]
        
        # Also consider alternative connections
        alt_prev = tour[(tour_idx - 2) % n]
        alt_next = tour[(tour_idx + 2) % n]
        
        alt_cost1 = distance_matrix[alt_prev, city] + distance_matrix[city, next_city]
        alt_cost2 = distance_matrix[prev_city, city] + distance_matrix[city, alt_next]
        
        best_alternative = min(alt_cost1, alt_cost2, direct_cost)
        return current_cost - best_alternative
    
    def compute_spatial_clustering(tour, coordinates, k_neighbors=8):
        n = len(tour)
        clustering_scores = np.zeros(n)
        
        for i in range(n):
            city = tour[i]
            point = coordinates[city]
            
            # Calculate distances to all other cities in tour
            tour_coords = coordinates[tour]
            distances = np.linalg.norm(tour_coords - point, axis=1)
            distances[i] = np.inf
            
            # Find k nearest neighbors in the current tour
            nearest_indices = np.argpartition(distances, k_neighbors)[:k_neighbors]
            avg_distance = np.mean(distances[nearest_indices])
            
            # Score is inverse of average distance (higher = more clustered)
            clustering_scores[i] = 1.0 / (avg_distance + 1e-10)
        
        return clustering_scores
    
    def adaptive_removal_strategy(tour, destroy_cnt, distance_matrix, coordinates):
        n = len(tour)
        removal_indices = []
        
        # Phase 1: Identify and remove from worst segments
        bad_segments = find_promising_removal_regions(tour, distance_matrix)
        segment_removals = min(destroy_cnt // 2, len(bad_segments) * 3)
        
        for start_idx in bad_segments[:segment_removals//3]:
            segment_size = min(3, destroy_cnt - len(removal_indices))
            for offset in range(segment_size):
                idx = (start_idx + offset) % n
                if idx not in removal_indices and len(removal_indices) < destroy_cnt:
                    removal_indices.append(idx)
        
        # Phase 2: Score-based removal for remaining slots
        remaining = destroy_cnt - len(removal_indices)
        if remaining > 0:
            scores = []
            for i in range(n):
                if i not in removal_indices:
                    # Improvement potential score
                    imp_score = calculate_local_improvement_potential(tour[i], i, tour, distance_matrix)
                    
                    # Spatial clustering score (if coordinates available)
                    cluster_score = 0
                    if coordinates is not None:
                        clustering_scores = compute_spatial_clustering(tour, coordinates)
                        cluster_score = clustering_scores[i]
                    
                    # Edge weight consideration
                    prev_edge = distance_matrix[tour[(i-1)%n], tour[i]]
                    next_edge = distance_matrix[tour[i], tour[(i+1)%n]]
                    edge_score = (prev_edge + next_edge) / 2
                    
                    # Composite score
                    total_score = imp_score * 0.5 + cluster_score * 0.3 + edge_score * 0.2
                    scores.append((total_score, i))
            
            scores.sort(reverse=True, key=lambda x: x[0])
            top_candidates = [idx for _, idx in scores[:remaining * 2]]
            
            if top_candidates:
                selected = random.sample(top_candidates, min(remaining, len(top_candidates)))
                removal_indices.extend(selected)
        
        # Phase 3: Fill any remaining slots randomly
        if len(removal_indices) < destroy_cnt:
            available = [i for i in range(n) if i not in removal_indices]
            if available:
                additional = random.sample(available, min(destroy_cnt - len(removal_indices), len(available)))
                removal_indices.extend(additional)
        
        return removal_indices[:destroy_cnt]
    
    actual_destroy_cnt = adaptive_destroy_size(destroy_cnt, len(x))
    
    if problem_data is not None and 'distance_matrix' in problem_data:
        distance_matrix = problem_data['distance_matrix']
        coordinates = problem_data.get('coordinates')
        tour = x
        
        # Use adaptive strategy based on problem size
        if len(tour) < 50:
            # For small problems, use more aggressive removal
            removal_indices = adaptive_removal_strategy(tour, actual_destroy_cnt, distance_matrix, coordinates)
        else:
            # For larger problems, use probabilistic strategy selection
            strategies = [
                lambda: adaptive_removal_strategy(tour, actual_destroy_cnt, distance_matrix, coordinates),
                lambda: random.sample(range(len(tour)), min(actual_destroy_cnt, len(tour))),
                lambda: list(np.random.choice(len(tour), min(actual_destroy_cnt, len(tour)), replace=False))
            ]
            weights = [0.6, 0.25, 0.15]
            strategy_idx = random.choices(range(len(strategies)), weights=weights)[0]
            removal_indices = strategies[strategy_idx]()
        
        # Ensure unique indices and sort for safe removal
        removal_indices = list(set(removal_indices))
        if len(removal_indices) > actual_destroy_cnt:
            removal_indices = random.sample(removal_indices, actual_destroy_cnt)
        removal_indices.sort(reverse=True)
        
        # Perform removal
        for idx in removal_indices:
            if 0 <= idx < len(new_x):
                removed_cities.append(new_x[idx])
                new_x.pop(idx)
    else:
        # Fallback for missing problem data
        start_idx = random.randint(0, len(new_x) - actual_destroy_cnt)
        for i in range(start_idx + actual_destroy_cnt - 1, start_idx - 1, -1):
            removed_cities.append(new_x[i])
            new_x.pop(i)
    
    return removed_cities, new_x
