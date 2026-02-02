import numpy as np
import random
import inspect
from typing import List, Any
import multiprocessing
from .instance import TSPInstance
from .get_instance import GetData
from .template import destroy_task_description, insert_task_description, destroy_template_program, insert_template_program
from .initial_operators import (
    destroy_v1, destroy_v2, destroy_v3,
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
            dist_mat = np.random.rand(n_cities, n_cities) * 100
            dist_mat = dist_mat.tolist()
            
            problem_data = {'distance_matrix': dist_mat}
            
            current_solution = list(range(n_cities))
            random.shuffle(current_solution)
            
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
                     
                if len(partial) != n_cities - destroy_cnt:
                    return_dict['result'] = (False, f"Partial length {len(partial)} != {n_cities - destroy_cnt}")
                    return
                if len(removed) != destroy_cnt:
                     return_dict['result'] = (False, f"Removed length {len(removed)} != {destroy_cnt}")
                     return
                
                combined = set(removed) | set(partial)
                if len(combined) != n_cities:
                     return_dict['result'] = (False, f"Lost or duplicated cities. Unique count: {len(combined)}")
                     return

            elif op_type == 'repair':
                temp_sol = list(current_solution)
                temp_removed = []
                for _ in range(destroy_cnt):
                    if temp_sol:
                        idx = random.randint(0, len(temp_sol)-1)
                        temp_removed.append(temp_sol.pop(idx))
                
                temp_partial = temp_sol
                
                try:
                    result = target_func(temp_partial, temp_removed, problem_data)
                except TypeError:
                    result = target_func(temp_partial, temp_removed)
                
                new_sol = result
                
                if not isinstance(new_sol, list):
                    return_dict['result'] = (False, "Did not return a list")
                    return

                if len(new_sol) != n_cities:
                    return_dict['result'] = (False, f"New solution length {len(new_sol)} != {n_cities}")
                    return
                if len(set(new_sol)) != n_cities:
                    return_dict['result'] = (False, "Duplicate/missing cities")
                    return

        return_dict['result'] = (True, None)

    except Exception as e:
        return_dict['result'] = (False, f"Execution Error in worker: {str(e)}")

class TSPTask:
    def __init__(self, n_instance: int = 10, n_cities: int = 50, seed: int = 2024):
        self.n_instance = n_instance
        self.n_cities = n_cities
        self.seed = seed
        self._destroy_task_description = destroy_task_description
        self._insert_task_description = insert_task_description
        self._destroy_template_program = destroy_template_program
        self._insert_template_program = insert_template_program
        self._load_instances()
        
        self.current_instance_idx = 0

    def _load_instances(self):
        data_loader = GetData(self.n_instance, self.n_cities)
        
        raw_instances = data_loader.generate_instances()
        
        self.instances = []
        for i, (coords, dist_mat) in enumerate(raw_instances):
            instance = TSPInstance(
                coordinates=coords,
                distance_matrix=dist_mat,
                name=f"Instance_{i}"
            )
            self.instances.append(instance)

    @property
    def current_instance(self) -> TSPInstance:
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
            'coordinates': inst.coordinates
        }

    def get_initial_solution(self) -> List[int]:
        sol = list(range(self.current_instance.num_cities))
        random.shuffle(sol)
        return sol

    def evaluate(self, solution: List[int]) -> float:
        dist_mat = self.current_instance.distance_matrix
        distance = 0.0
        n = len(solution)
        
        for i in range(n - 1):
            distance += dist_mat[solution[i]][solution[i + 1]]
        
        if n > 0:
            distance += dist_mat[solution[-1]][solution[0]]
            
        return distance

    def write_solution(self, solution: List[int], filepath: str):
        distance = self.evaluate(solution)
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, route in enumerate(solution):
                f.write(f"Route #{i+1}: " + " ".join(map(str, route)) + "\n")
            f.write(f"Distance: {distance}\n")

    def get_initial_operators(self):
        seeds_destroy = [
            {'name': 'random_destroy', 'func': destroy_v1, 'code': inspect.getsource(destroy_v1), 'score': 0.0, 'weight': 1.0},
            {'name': 'max_dist_destroy', 'func': destroy_v2, 'code': inspect.getsource(destroy_v2), 'score': 0.0, 'weight': 1.0},
            {'name': 'continue_destroy', 'func': destroy_v3, 'code': inspect.getsource(destroy_v3), 'score': 0.0, 'weight': 1.0}
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
