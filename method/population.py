import random
import numpy as np
from typing import List, Dict, Any, Optional

class Population:
    def __init__(self, pop_size: int = 5):
        self._pop_size = pop_size
        self.destroy_operators: List[Dict[str, Any]] = []
        self.repair_operators: List[Dict[str, Any]] = []

    def add_destroy_operator(self, operator: Dict[str, Any]):
        self.destroy_operators.append(operator)

    def add_repair_operator(self, operator: Dict[str, Any]):
        self.repair_operators.append(operator)

    def get_destroy_operators(self) -> List[Dict[str, Any]]:
        return self.destroy_operators

    def get_repair_operators(self) -> List[Dict[str, Any]]:
        return self.repair_operators

    def prune(self, prune_size: int = 2):
        self._prune_list(self.destroy_operators, "Destroy", prune_size)
        self._prune_list(self.repair_operators, "Repair", prune_size)

    def _prune_list(self, operators: List[Dict[str, Any]], op_type: str, prune_size: int = 2):
        op_scores_rank = sorted(operators, key=lambda x: x['score'], reverse=True)
        n_keep = max(0, len(operators) - prune_size)
        kept_operators = op_scores_rank[:n_keep]
        removed_operators = op_scores_rank[n_keep:]
        if removed_operators:
            print(f"[Population] Pruned {op_type} Operators: {[op['name'] for op in removed_operators]}")
        operators[:] = kept_operators

    def reset_scores(self):
        print(f"[Population] Resetting scores for next epoch.")
        for op in self.destroy_operators + self.repair_operators:
            op['score'] = 0.0
            
    def get_best_operator(self, op_type: str) -> Optional[Dict]:
        ops = self.destroy_operators if op_type == 'destroy' else self.repair_operators
        if not ops: return None
        return max(ops, key=lambda x: x.get('score', 0.0))
        
    def get_worst_operator(self, op_type: str) -> Optional[Dict]:
        ops = self.destroy_operators if op_type == 'destroy' else self.repair_operators
        if not ops: return None
        return min(ops, key=lambda x: x.get('score', 0.0))

    def update_scores(self, destroy_scores: Dict[str, float], repair_scores: Dict[str, float]):
        for op in self.destroy_operators:
            if op['name'] in destroy_scores:
                op['score'] += destroy_scores[op['name']]
        
        for op in self.repair_operators:
            if op['name'] in repair_scores:
                op['score'] += repair_scores[op['name']]

    def select_parents(self, operators: List[Dict[str, Any]], n_parents: int = 2) -> List[Dict[str, Any]]:
        if len(operators) < n_parents:
            return random.choices(operators, k=n_parents)
        if n_parents == 1:
            return [random.choice(operators)]
        else:
            scores = np.array([max(0, op['score']) for op in operators])
            total_score = np.sum(scores)
            
            if total_score == 0:
                probs = np.ones(len(operators)) / len(operators)
            else:
                probs = scores / total_score
                
            parent_indices = np.random.choice(len(operators), size=n_parents, p=probs, replace=False)
            return [operators[i] for i in parent_indices]