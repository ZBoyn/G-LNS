from typing import Optional, Dict, Any, Tuple, Callable, List
import numpy as np
import random
import copy
import inspect
import re
import time
import logging
from method.population import Population
from tools.profiler import ProfilerBase
import os
from task.taskBase import TaskBase

class GLNSUtil:
    @staticmethod
    def init_op_logger(profiler: Optional[ProfilerBase] = None):
        logger = logging.getLogger('glns_operators')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            log_dir = './logs/glns'
            if profiler and profiler._log_dir:
                log_dir = profiler._log_dir
            
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, 'operators_history.log'), mode='a', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        return logger

    @staticmethod
    def save_solution(output_dir: str, task: TaskBase, solution):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = f"{task.current_instance.name}.sol"
        filepath = os.path.join(output_dir, filename)
        if hasattr(task, 'write_solution'):
            task.write_solution(solution, filepath)

    @staticmethod
    def exec_code_str(code_str: str) -> Dict[str, Any]:
        g = {
            'np': np,
            'random': random,
            'List': List,
            'Dict': Dict,
            'Any': Any,
            'Tuple': Tuple,
            'copy': copy,
        }
        exec(code_str, g)
        return g

    @staticmethod
    def extract_operators(g: Dict[str, Any]) -> Tuple[Optional[Callable], Optional[Callable]]:
        d_func = g.get('destroy_operator')
        r_func = g.get('repair_operator')

        if not d_func or not r_func:
            candidates = [v for k, v in g.items() if callable(v) and k not in ['np', 'random', 'List', 'Dict', 'Any', 'Tuple']]
            
            for func in candidates:
                try:
                    sig = inspect.signature(func)
                    params = sig.parameters
                    if d_func is None and any(p in params for p in ['destroy_cnt']):
                        d_func = func
                    elif r_func is None and any(p in params for p in ['removed_customers', 'removed_cities']):                        
                        r_func = func
                except ValueError:
                    continue
        
        return d_func, r_func

    @staticmethod
    def create_operator_dict(func: Callable, code_str: str, source_info: str = "") -> Dict[str, Any]:
        unique_name = f"{func.__name__}_{int(time.time())}_{random.randint(100,999)}"
        
        pattern = r"(?<!\w)" + re.escape(func.__name__) + r"(?!\w)"
        new_code = re.sub(pattern, unique_name, code_str)

        new_op = {
            'name': unique_name,
            'func': func,
            'code': new_code, 
            'score': 0.0, 
            'weight': 1.0, 
        }
        return new_op

    @staticmethod
    def compile_and_register_joint_pair(
        code_str: str, 
        source_info: str,
        task: TaskBase,
        population: Population,
        op_logger: logging.Logger,
        check_timeout: float
    ) -> Tuple[bool, Optional[str]]:
        try:
            g = GLNSUtil.exec_code_str(code_str)
            d_func, r_func = GLNSUtil.extract_operators(g)
            
            if not d_func or not r_func:
                msg = "Failed to identify both Destroy and Repair functions in joint code."
                print(f"[LLM] {msg}")
                return False, msg

            if not task.check_operator(d_func, code_str, 'destroy', timeout=check_timeout):
                msg = f"Destroy operator {d_func.__name__} failed check."
                print(f"[LLM] {msg}")
                return False, msg

            if not task.check_operator(r_func, code_str, 'repair', timeout=check_timeout):
                msg = f"Repair operator {r_func.__name__} failed check."
                print(f"[LLM] {msg}")
                return False, msg

            new_op_d = GLNSUtil.create_operator_dict(d_func, code_str, source_info)
            population.add_destroy_operator(new_op_d)
            op_logger.info(f"[DESTROY] Joint Registered: {new_op_d['name']}")
            print(f"[LLM] Joint Registered Destroy: {new_op_d['name']}")

            new_op_r = GLNSUtil.create_operator_dict(r_func, code_str, source_info)
            population.add_repair_operator(new_op_r)
            op_logger.info(f"[REPAIR] Joint Registered: {new_op_r['name']}")
            print(f"[LLM] Joint Registered Repair: {new_op_r['name']}")
            
            return True, None

        except Exception as e:
            msg = f"Failed to compile/register joint operators: {e}"
            print(f"[LLM] {msg}")
            return False, str(e)

    @staticmethod
    def compile_and_register_operator(
        code_str: str, 
        op_type: str, 
        source_info: str,
        task: TaskBase,
        population: Population,
        op_logger: logging.Logger,
        check_timeout: float,
        debug_mode: bool = False
    ) -> Tuple[bool, Optional[str]]:
        try:
            g = GLNSUtil.exec_code_str(code_str)
            d_func, r_func = GLNSUtil.extract_operators(g)
            
            target_func = d_func if op_type == 'destroy' else r_func
            
            if not target_func:
                msg = f"Could not find valid {op_type} function in generated code."
                print(f"[LLM] {msg}")
                return False, msg

            if debug_mode:
                print(f"[LLM] Compiled operator: {target_func.__name__}")

            if not task.check_operator(target_func, code_str, op_type, timeout=check_timeout):
                msg = f"Operator {target_func.__name__} failed check."
                print(f"[LLM] {msg}")
                return False, msg
            
            new_op = GLNSUtil.create_operator_dict(target_func, code_str, source_info)
            
            if op_type == 'destroy':
                population.add_destroy_operator(new_op)
            else:
                population.add_repair_operator(new_op)
            
            op_logger.info(f"[{op_type.upper()}] Registered: {new_op['name']} | Source: {source_info}")
            op_logger.info(f"Code:\n{code_str}\n{'-'*50}")
            
            print(f"[LLM] Successfully registered new {op_type} operator: {new_op['name']}")
            return True, None
            
        except Exception as e:
            msg = f"Failed to compile/register operator: {e}"
            print(f"[LLM] {msg}")
            if debug_mode:
                pass
            return False, str(e)