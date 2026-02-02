from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Callable

class TaskBase(ABC):
    def __init__(self, name: str):
        self.name = name

        self._destroy_template_program: str = ""
        self._insert_template_program: str = ""
        self._destroy_task_description: str = ""
        self._insert_task_description: str = ""

        self.n_instance: int = 0

    @abstractmethod
    def set_current_instance(self, instance_idx: int) -> None:
        pass

    @abstractmethod
    def get_initial_solution(self) -> Any:
        pass

    @abstractmethod
    def evaluate(self, solution: Any) -> float:
        pass

    @abstractmethod
    def get_problem_data(self) -> Any:
        pass

    @abstractmethod
    def get_initial_operators(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        pass

    @abstractmethod
    def check_operator(self, func: Callable, code_str: str, op_type: str, timeout: float = 10.0) -> bool:
        pass

    @abstractmethod
    def write_solution(self, solution: Any, filepath: str) -> None:
        pass