import random
import os
from typing import Callable, Tuple, Optional

from base import LLM
from tools.profiler import ProfilerBase
from .prompt import GLNSPrompt
from .population import Population
from .evaluate import ALNSEvaluator
from .sampler import GLNSSampler
from .util import GLNSUtil
from task.taskBase import TaskBase

class GLNS:
    def __init__(self,
                 llm: LLM,
                 task: TaskBase,
                 profiler: ProfilerBase = None,
                 max_generations: int = 10,
                 pop_size: int = 5,
                 prune_size: int = 2,
                 check_timeout: float = 10.0,
                 *,
                 init_with_llm: bool = True,
                 debug_mode: bool = False,
                 **kwargs):
        
        self._llm = llm
        self._profiler = profiler
        self._max_generations = max_generations
        self._pop_size = pop_size
        self._debug_mode = debug_mode
        self._init_with_llm = init_with_llm
        self._prune_size = prune_size
        self._check_timeout = check_timeout

        # Initialize Logger
        self._op_logger = GLNSUtil.init_op_logger(profiler=profiler)
        
        # Initialize Task
        self._destroy_template_program = task._destroy_template_program
        self._insert_template_program = task._insert_template_program
        self._destroy_task_description = task._destroy_task_description
        self._insert_task_description = task._insert_task_description
        
        self._task = task

        # Initialize Sampler
        self._sampler = GLNSSampler(llm)

        # Initialize Population
        self.population = Population(self._pop_size)
        self._init_seed_operators()

        # Initialize Evaluator
        self.alns_evaluator = ALNSEvaluator(self._task, debug_mode=debug_mode, **kwargs)

        self._current_generation = 0
        self._pair_scores = {}

    def _init_seed_operators(self):
        if self._init_with_llm:
            seeds_destroy, seeds_repair = self._task.get_initial_operators()
            
            for i in range(self._pop_size - len(seeds_destroy)):
                prompt = GLNSPrompt.destroy_initialization_prompt(self._destroy_task_description, seeds_destroy)
                
                self._generate_with_retry(
                    prompt,
                    lambda code: GLNSUtil.compile_and_register_operator(
                        code, 'destroy', f"Seed Destroy Operator {i + 1}", 
                        self._task, self.population, self._op_logger, 
                        self._check_timeout, self._debug_mode
                    )
                )

            for i in range(self._pop_size - len(seeds_repair)):
                prompt = GLNSPrompt.repair_initialization_prompt(self._insert_task_description, seeds_repair)
                
                self._generate_with_retry(
                    prompt,
                    lambda code: GLNSUtil.compile_and_register_operator(
                        code, 'repair', f"Seed Repair Operator {i + 1}", 
                        self._task, self.population, self._op_logger, 
                        self._check_timeout, self._debug_mode
                    )
                )

        else:
            seeds_destroy, seeds_repair = self._task.get_initial_operators()

        for op in seeds_destroy:
            self.population.add_destroy_operator(op)
            
        for op in seeds_repair:
            self.population.add_repair_operator(op)

    def _evolve_operators(self):
        print(f"[Evolution] Generation {self._current_generation}: Evolving Operators (LLM)")
        for _ in range(self._prune_size):
            if random.random() < 0.5:
                self._evolve_mutation('destroy')
                self._evolve_mutation('repair')
            elif random.random() < 0.8:
                self._evolve_homogeneous_crossover('destroy')
                self._evolve_homogeneous_crossover('repair')
            else:
                self._evolve_synergistic_joint_crossover()

        self.population.reset_scores()

    def _evolve_mutation(self, op_type: str):
        if op_type == 'destroy':
            ops = self.population.get_destroy_operators()
            task_desc = self._destroy_task_description
        else:
            ops = self.population.get_repair_operators()
            task_desc = self._insert_task_description

        parents = self.population.select_parents(ops, n_parents=1)

        if parents:
            op = parents[0]
            if self._debug_mode:
                print(f"[LLM] Mutating {op_type}: {op['name']}")
            sorted_ops = sorted(ops, key=lambda x: x.get('score', 0), reverse=True)
            rank = sorted_ops.index(op)
            if rank == 0:
                strategy = 'best'
            elif rank == len(sorted_ops) - 1:
                strategy = 'worst'
            else:
                strategy = 'random'
            
            prompt = GLNSPrompt.mutation_prompt(op['code'], strategy, op_type, task_description=task_desc)
            
            self._generate_with_retry(
                prompt,
                lambda code: GLNSUtil.compile_and_register_operator(
                    code, op_type, f"Mutation ({strategy}) from {op['name']}",
                    self._task, self.population, self._op_logger,
                    self._check_timeout, self._debug_mode
                )
            )

    def _evolve_homogeneous_crossover(self, op_type: str):
        if op_type == 'destroy':
            ops = self.population.get_destroy_operators()
            task_desc = self._destroy_task_description
        else:
            ops = self.population.get_repair_operators()
            task_desc = self._insert_task_description

        if len(ops) < 2:
            if self._debug_mode:
                print(f"[LLM] Not enough operators to perform homogeneous crossover {op_type}.")
            return

        parents = self.population.select_parents(ops, n_parents=2)

        if parents:
            if self._debug_mode:
                print(f"[LLM] Homogeneous Crossover {op_type}: {parents[0]['name']} + {parents[1]['name']}")
                
            prompt = GLNSPrompt.homogeneous_crossover_prompt(parents[0]['code'], parents[1]['code'], op_type, task_description=task_desc)
            
            self._generate_with_retry(
                prompt,
                lambda code: GLNSUtil.compile_and_register_operator(
                    code, op_type, f"Crossover {parents[0]['name']} + {parents[1]['name']}",
                    self._task, self.population, self._op_logger,
                    self._check_timeout, self._debug_mode
                )
            )

    def _evolve_synergistic_joint_crossover(self):
        if not self._pair_scores:
            if self._debug_mode: print("[LLM] No pair scores available for Synergistic Joint Crossover.")
            return

        best_pair_key = max(self._pair_scores, key=self._pair_scores.get)
        try:
            d_name, r_name = best_pair_key.split('|')
        except ValueError:
            return 

        d_ops = {op['name']: op for op in self.population.get_destroy_operators()}
        r_ops = {op['name']: op for op in self.population.get_repair_operators()}

        if d_name in d_ops and r_name in r_ops:
            d_op = d_ops[d_name]
            r_op = r_ops[r_name]
            
            if self._debug_mode:
                print(f"[LLM] Synergistic Joint Crossover: {d_name} + {r_name}")
            
            task_desc = f"Destroy Task:\n{self._destroy_task_description}\n\nRepair Task:\n{self._insert_task_description}"
            
            prompt = GLNSPrompt.synergistic_joint_crossover_prompt(d_op['code'], r_op['code'], task_description=task_desc)
            
            self._generate_with_retry(
                prompt,
                lambda code: GLNSUtil.compile_and_register_joint_pair(
                    code, f"Synergistic Joint Crossover {d_name}+{r_name}",
                    self._task, self.population, self._op_logger,
                    self._check_timeout
                )
            )
    
    def _generate_with_retry(self, 
                             prompt: str, 
                             register_callback: Callable[[str], Tuple[bool, Optional[str]]],
                             max_retries: int = 3) -> bool:
        current_prompt = prompt
        for i in range(max_retries + 1):
            if i > 0:
                print(f"[LLM] Retry attempt {i}/{max_retries}...")
                
            code_str, raw_response = self._sampler.get_code(current_prompt, return_raw=True)
            if not code_str:
                error_msg = "No code block found in response."
                print(f"[LLM] Generation Error: {error_msg}")
                print(f"[LLM] Raw Response BEGIN:\n{raw_response}\n[LLM] Raw Response END")
                current_prompt += f"\n\nError: {error_msg}\nPlease ensure you wrap the code in ```python ... ``` block."
                continue

            success, error_msg = register_callback(code_str)
            if success:
                return True
            
            current_prompt += f"\n\nThe previous code you generated had an error:\n{error_msg}\n\nPlease fix the code and try again. Ensure it follows all constraints."
        
        print(f"[LLM] Failed after {max_retries} retries.")
        return False

    def run(self):
        print("Starting G-LNS Framework...")

        while self._current_generation < self._max_generations:
            print(f"\n=== Generation {self._current_generation} ===")
            best_f, d_scores, r_scores, pair_scores, best_sol = self.alns_evaluator.evaluate(self.population, initial_solution=None)
            self._pair_scores = pair_scores
            print(f"Best Solution: {best_f:.2f}")

            with open(os.path.join(self._profiler._log_dir, 'best_solution.txt'), 'a', encoding='utf-8') as f:
                f.write(f"=== Generation {self._current_generation} ===\n")
                f.write(f"Best Solution: {best_f:.2f}\n")
            
            self.population.update_scores(d_scores, r_scores)

            self._op_logger.info(f"=== Gen {self._current_generation} Scores ===")
            self._op_logger.info(f"Destroy Scores: {d_scores}")
            self._op_logger.info(f"Repair Scores: {r_scores}")

            if (self._current_generation + 1) % 10 == 0:
                print(f"[Evolution] Triggering evolution at generation {self._current_generation}")
                
                self.population.prune(self._prune_size)
                self._evolve_operators()
            
            self._current_generation += 1

        for _ in range(len(self.population.get_destroy_operators())):
            self._op_logger.info(f"Destroy Operator {_ + 1}: {self.population.get_destroy_operators()[_]['name']} | Score: {self.population.get_destroy_operators()[_]['score']}")
        for _ in range(len(self.population.get_repair_operators())):
            self._op_logger.info(f"Repair Operator {_ + 1}: {self.population.get_repair_operators()[_]['name']} | Score: {self.population.get_repair_operators()[_]['score']}")

        print("G-LNS Finished.")

        if self._profiler:
            self._profiler.finish()