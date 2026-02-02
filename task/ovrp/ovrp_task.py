import numpy as np
import random
import inspect
from typing import List, Any
import multiprocessing
from .instance import OVRPInstance
from .get_instance import GetData
from .template import destroy_task_description, insert_task_description, destroy_template_program, insert_template_program
from .initial_operators import (
    destroy_v1, destroy_v2,
    insert_v1, insert_v2
)
from method.util import GLNSUtil

def _worker_execute_test(code_str: str, op_type: str, test_cases: List[tuple], return_dict):
    try:
        g = GLNSUtil.exec_code_str(code_str)
        d_func, r_func = GLNSUtil.extract_operators(g)
        target_func = d_func if op_type == 'destroy' else r_func
        
        if not target_func:
             return_dict['result'] = (False, f"Could not extract {op_type} function from code.")
             return

        for n_cities, destroy_cnt in test_cases:
            n_nodes = n_cities + 1
            depot_idx = 0
            
            dist_mat = np.random.rand(n_nodes, n_nodes) * 100
            coords = np.random.rand(n_nodes, 2)
            
            demands = np.ones(n_nodes, dtype=int)
            demands[0] = 0 
            
            capacity = max(10, n_cities // 2)
            
            problem_data = {
                'distance_matrix': dist_mat,
                'demands': demands,
                'capacity': capacity,
                'depot_idx': depot_idx,
                'coordinates': coords
            }
            
            customers = list(range(1, n_nodes))
            random.shuffle(customers)
            current_solution = []
            chunk_size = max(1, capacity // 2)
            for i in range(0, len(customers), chunk_size):
                current_solution.append(customers[i:i + chunk_size])
                
            if op_type == 'destroy':
                try:
                    result = target_func(current_solution, destroy_cnt, problem_data)
                except TypeError:
                    result = target_func(current_solution, destroy_cnt)
                    
                if not isinstance(result, (tuple, list)) or len(result) != 2:
                    return_dict['result'] = (False, f"Invalid return format: {type(result)}")
                    return
                
                removed, partial = result
                
                if not isinstance(removed, list) or not isinstance(partial, list):
                     return_dict['result'] = (False, "Return values must be lists.")
                     return

                flat_partial = [c for r in partial for c in r]
                if len(flat_partial) != n_cities - destroy_cnt:
                        return_dict['result'] = (False, f"Partial length {len(flat_partial)} != {n_cities - destroy_cnt}")
                        return
                if len(removed) != destroy_cnt:
                        return_dict['result'] = (False, f"Removed length {len(removed)} != {destroy_cnt}")
                        return
                
                combined = set(removed) | set(flat_partial)
                if len(combined) != n_cities:
                        return_dict['result'] = (False, f"Lost or duplicated cities. Unique count: {len(combined)}")
                        return

            elif op_type == 'repair':
                temp_sol = [list(r) for r in current_solution]
                
                all_customers_in_sol = [c for r in temp_sol for c in r]
                if len(all_customers_in_sol) < destroy_cnt:
                     continue

                to_remove_indices = random.sample(range(len(all_customers_in_sol)), destroy_cnt)
                to_remove_values = [all_customers_in_sol[i] for i in to_remove_indices]
                
                temp_removed = to_remove_values
                
                new_temp_sol = []
                for route in temp_sol:
                    new_route = [c for c in route if c not in to_remove_values]
                    if new_route:
                        new_temp_sol.append(new_route)
                
                temp_partial = new_temp_sol
                
                try:
                    result = target_func(temp_partial, temp_removed, problem_data)
                except TypeError:
                    result = target_func(temp_partial, temp_removed)

                new_sol = result
                
                if not isinstance(new_sol, list):
                    return_dict['result'] = (False, "Did not return a list")
                    return

                flat_new = [c for r in new_sol for c in r]
                if len(flat_new) != n_cities:
                    return_dict['result'] = (False, f"New solution length {len(flat_new)} != {n_cities}")
                    return
                if len(set(flat_new)) != n_cities:
                    return_dict['result'] = (False, "Duplicate/missing cities")
                    return

        return_dict['result'] = (True, None)

    except Exception as e:
        return_dict['result'] = (False, f"Execution Error in worker: {str(e)}")


class OVRPTask:
    def __init__(self, n_instance: int = 10, n_cities: int = 50, capacity: int = 50, seed: int = 2024):
        self.n_instance = n_instance
        self.n_cities = n_cities
        self.capacity = capacity
        self.seed = seed
        self._destroy_task_description = destroy_task_description
        self._insert_task_description = insert_task_description
        self._destroy_template_program = destroy_template_program
        self._insert_template_program = insert_template_program
        self._load_instances()
        
        self.current_instance_idx = 0

    def _load_instances(self):
        data_loader = GetData(self.n_instance, self.n_cities, self.capacity)
        
        raw_instances = data_loader.generate_instances()
        
        self.instances = []
        for i, (coords, dist_mat, demands, cap) in enumerate(raw_instances):
            demands[0] = 0
            instance = OVRPInstance(
                coordinates=coords,
                distance_matrix=dist_mat,
                demands=demands,
                capacity=cap,
                name=f"OVRP_Instance_{i}"
            )
            self.instances.append(instance)

    @property
    def current_instance(self) -> OVRPInstance:
        return self.instances[self.current_instance_idx]

    def set_current_instance(self, idx: int):
        if 0 <= idx < len(self.instances):
            self.current_instance_idx = idx
        else:
            raise IndexError(f"[Error] Instance index {idx} out of range.")

    def get_problem_data(self) -> dict:
        inst = self.current_instance
        return {
            'distance_matrix': inst.distance_matrix,
            'demands': inst.demands,
            'capacity': inst.capacity,
            'depot_idx': inst.depot_idx,
            'coordinates': inst.coordinates
        }

    def get_initial_solution(self) -> List[List[int]]:
        inst = self.current_instance
        demands = inst.demands
        capacity = inst.capacity
        
        customers = list(range(1, inst.num_nodes))
        random.shuffle(customers)
        
        routes = []
        current_route = []
        current_load = 0
        
        for c in customers:
            if current_load + demands[c] <= capacity:
                current_route.append(c)
                current_load += demands[c]
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [c]
                current_load = demands[c]
        
        if current_route:
            routes.append(current_route)
            
        return routes

    def evaluate(self, solution: List[List[int]]) -> float:
        inst = self.current_instance
        dist_mat = inst.distance_matrix
        demands = inst.demands
        capacity = inst.capacity
        depot = inst.depot_idx
        
        total_dist = 0.0
        
        all_visited = set()
        
        for route in solution:
            if not route: continue
            
            route_load = sum(demands[node] for node in route)
            if route_load > capacity:
                return float('inf')

            total_dist += dist_mat[depot][route[0]]

            for i in range(len(route) - 1):
                total_dist += dist_mat[route[i]][route[i+1]]
            
            # OVRP: No return to depot
            
            for node in route:
                all_visited.add(node)
                
        if len(all_visited) != inst.num_nodes - 1:
             return float('inf')
            
        return total_dist

    def write_solution(self, solution: List[List[int]], filepath: str):
        cost = self.evaluate(solution)
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, route in enumerate(solution):
                f.write(f"Route #{i+1}: " + " ".join(map(str, route)) + "\n")
            f.write(f"Cost {int(cost) if cost != float('inf') else 'Inf'}\n")

    def get_initial_operators(self):
        seeds_destroy = [
            {'name': 'random_destroy', 'func': destroy_v1, 'code': inspect.getsource(destroy_v1), 'score': 0.0, 'weight': 1.0},
            {'name': 'worst_cost_destroy', 'func': destroy_v2, 'code': inspect.getsource(destroy_v2), 'score': 0.0, 'weight': 1.0},
        ]
        seeds_repair = [
            {'name': 'random_insert', 'func': insert_v1, 'code': inspect.getsource(insert_v1), 'score': 0.0, 'weight': 1.0},
            {'name': 'greedy_insert', 'func': insert_v2, 'code': inspect.getsource(insert_v2), 'score': 0.0, 'weight': 1.0}
        ]
        return seeds_destroy, seeds_repair

    def check_operator(self, func_callable, code_str: str, op_type: str, timeout: float = 10.0) -> bool:
        test_cases = [
            (10, 2),
            (20, 5),
            (10, 1)
        ]

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        
        p = multiprocessing.Process(target=_worker_execute_test, args=(code_str, op_type, test_cases, return_dict))
        p.start()
        
        p.join(timeout)
        
        if p.is_alive():
            print(f"[Error] Check failed: Operator timed out (>{timeout}s). Terminating process.")
            p.terminate()
            p.join()
            return False
        
        if 'result' not in return_dict:
            print(f"[Error] Check failed: Worker process crashed or returned no result.")
            return False
            
        success, msg = return_dict['result']
        if not success:
            print(f"[Error] {msg}")
            return False
                
        return True
