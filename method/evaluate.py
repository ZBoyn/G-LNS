import numpy as np
import os
import copy
from typing import List, Dict, Any, Tuple
from .population import Population
from tqdm import tqdm
from .util import GLNSUtil

class ALNSEvaluator:
    def __init__(self, task, debug_mode: bool = False, save_solutions: bool = False, output_dir: str = None, **kwargs):
        self.task = task
        self.debug_mode = debug_mode
        self.save_solutions = save_solutions
        self.output_dir = output_dir
        
        # ALNS Hyperparameters
        self.T_start = kwargs.get('T_start', 100)
        self.alpha = kwargs.get('alpha', 0.97)
        self.max_iter = kwargs.get('max_iter', 100)
        self.lambda_rate = kwargs.get('lambda_rate', 0.5)
        self.destroy_rate = kwargs.get('destroy_rate', 0.2)
        self.use_tqdm = kwargs.get('use_tqdm', False)
        self.op_timeout = kwargs.get('op_timeout', 10.0)

        # ALNS Score Rewards
        self.reward_sigma1 = kwargs.get('reward_sigma1', 1.5)
        self.reward_sigma2 = kwargs.get('reward_sigma2', 1.2)
        self.reward_sigma3 = kwargs.get('reward_sigma3', 0.8)
        self.reward_sigma4 = kwargs.get('reward_sigma4', 0.1)

    def _select_operator(self, operators: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        weights = [op['weight'] for op in operators]
        total_weight = sum(weights)
        
        if total_weight <= 0:
            probs = [1.0/len(operators)] * len(operators)
        else:
            probs = [w/total_weight for w in weights]
        
        idx = np.random.choice(range(len(operators)), p=probs)
        return idx, operators[idx]

    def evaluate(self, population: Population, initial_solution: List[int] = None) -> Tuple[float, Dict[str, float], Dict[str, float], List[int], Dict[str, float]]:
        """
        Run the Evaluation main loop across ALL instances.
        Returns:
            - avg_best_f: The average best objective value found across all instances.
            - destroy_scores: Accumulated scores for destroy operators in this run.
            - repair_scores: Accumulated scores for repair operators in this run.
            - pair_scores: Accumulated scores for (Destroy, Repair) pair operators in this run.
            - best_solution: The best solution found in the LAST instance.
        """
        destroy_ops = population.get_destroy_operators()
        repair_ops = population.get_repair_operators()
        
        for op in destroy_ops: op['weight'] = 1.0
        for op in repair_ops: op['weight'] = 1.0
        
        if not destroy_ops or not repair_ops:
            print("[ALNS] Warning: Empty operator pool.")
            return float('inf'), {}, {}, []

        total_best_f = 0.0
        global_destroy_scores = {op['name']: 0.0 for op in destroy_ops}
        global_repair_scores = {op['name']: 0.0 for op in repair_ops}
        global_pair_scores = {} # Key: f"{d_name}|{r_name}", Value: score
        
        last_best_solution = None

        n_instances = len(self.task.instances)
        
        obj_curves = []
        
        for instance_idx in range(n_instances):
            self.task.set_current_instance(instance_idx)
            
            destroy_scores_acc = {op['name']: 0.0 for op in destroy_ops}
            repair_scores_acc = {op['name']: 0.0 for op in repair_ops}
            pair_scores_acc = {}
            obj_curves_ins = []

            if initial_solution is not None:
                 current_x = copy.deepcopy(initial_solution)
            else:
                 current_x = self.task.get_initial_solution()

            current_f = self.task.evaluate(current_x)
            
            best_x = copy.deepcopy(current_x)
            best_f = current_f
            
            T = self.T_start
            cur_iter = 0
            
            problem_data = self.task.get_problem_data()
            
            pbar = None
            if self.use_tqdm:
                pbar = tqdm(total=self.max_iter, desc=f"Evaluating Instance {instance_idx + 1}/{n_instances}", leave=False)
                
            try:
                while cur_iter < self.max_iter:
                    d_idx, d_op = self._select_operator(destroy_ops)
                    r_idx, r_op = self._select_operator(repair_ops)
                    
                    if isinstance(current_x[0], list):
                        n_size = sum(len(r) for r in current_x)
                    else: 
                        n_size = len(current_x)
                    
                    destroy_cnt = int(n_size * self.destroy_rate)

                    x_temp = copy.deepcopy(current_x)
                    
                    op_failed = False
                    try:
                        removed_elements, partial_solution = d_op['func'](x_temp, destroy_cnt, problem_data)
                        new_x = r_op['func'](partial_solution, removed_elements, problem_data)
                        
                        if isinstance(new_x[0], list):
                            new_size = sum(len(r) for r in new_x)
                            old_size = sum(len(r) for r in current_x)
                        else:
                            new_size = len(new_x)
                            old_size = len(current_x)

                        if new_size != old_size:
                            raise ValueError(f"Solution size mismatch: {new_size} vs {old_size}")
                            
                    except Exception as e:
                        if self.debug_mode:
                            print(f"Error in operators {d_op['name']} / {r_op['name']}: {e}")
                        new_x = current_x
                        op_failed = True
                    
                    if op_failed:
                        score_incre = self.reward_sigma4
                        new_f = current_f
                    else:
                        new_f = self.task.evaluate(new_x)
                        
                        if new_f == float('inf'):
                            score_incre = self.reward_sigma4
                        else:
                            score_incre = 0.0
                            
                            if new_f < current_f:
                                current_x = new_x
                                current_f = new_f
                                if new_f < best_f:
                                    best_x = new_x
                                    best_f = new_f
                                    score_incre = self.reward_sigma1
                                else:
                                    score_incre = self.reward_sigma2
                            elif new_f == current_f:
                                 score_incre = self.reward_sigma4
                            else:
                                delta = new_f - current_f
                                prob = np.exp(-delta / T)
                                     
                                if np.random.random() < prob:
                                    current_x = new_x
                                    current_f = new_f
                                    score_incre = self.reward_sigma3
                                else:
                                    score_incre = self.reward_sigma4
                    
                    obj_curves_ins.append(best_f)
                
                    d_op['weight'] = d_op['weight'] * self.lambda_rate + (1 - self.lambda_rate) * score_incre
                    r_op['weight'] = r_op['weight'] * self.lambda_rate + (1 - self.lambda_rate) * score_incre
                    
                    destroy_scores_acc[d_op['name']] += score_incre
                    repair_scores_acc[r_op['name']] += score_incre
                    
                    pair_key = f"{d_op['name']}|{r_op['name']}"
                    if pair_key not in pair_scores_acc:
                        pair_scores_acc[pair_key] = 0.0
                    pair_scores_acc[pair_key] += score_incre
                    
                    T = T * self.alpha
                    cur_iter += 1

                    if pbar:
                        pbar.update(1)
            finally:
                if pbar:
                    pbar.close()
            
            obj_curves.append(obj_curves_ins)

            total_best_f += best_f

            if self.save_solutions and self.output_dir:
                GLNSUtil.save_solution(self.output_dir, self.task, best_x)

            for name, score in destroy_scores_acc.items():
                global_destroy_scores[name] += score
            for name, score in repair_scores_acc.items():
                global_repair_scores[name] += score
            for key, score in pair_scores_acc.items():
                if key not in global_pair_scores:
                    global_pair_scores[key] = 0.0
                global_pair_scores[key] += score
            
            last_best_solution = best_x
                
        avg_best_f = total_best_f / n_instances
        
        if obj_curves and self.save_solutions:
            avg_obj_curve = []
            for i in range(len(obj_curves[0])):
                val = sum(c[i] for c in obj_curves) / n_instances
                avg_obj_curve.append(val)
            np.savetxt(os.path.join(self.output_dir, "convergence_avg.txt"), avg_obj_curve)

        return avg_best_f, global_destroy_scores, global_repair_scores, global_pair_scores, last_best_solution