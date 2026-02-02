import random

class GLNSPrompt:
    
    @staticmethod
    def destroy_initialization_prompt(task_description: str, existing_operators: list):
        operators_str = "\n".join([f"- {op['name']}: {op['code']}" for op in existing_operators])
        
        prompt = f"""
You are an expert in heuristic optimization algorithms, specifically Large Neighborhood Search (LNS).
Your task is to design a new 'Destroy Operator' (removal operator) for the following problem:

Problem Description:
{task_description}

Existing Destroy Operators (Reference):
{operators_str}

    Requirement:
    1. First, describe your new algorithm and main steps in one sentence. The description must be inside a brace. Next, implement it in Python as a function named destroy.
    2. The logic should be strictly different from the existing ones provided in the reference to improve population diversity.
    3. Do not give additional explanations.
    4. IMPORTANT: Wrap the code in a Markdown code block (e.g., ```python ... ```).
    5. Do not assume any external functions exist other than standard Python libraries and what is provided in problem_data.
    6. Do NOT evaluate numpy arrays as booleans (e.g. `if demands:`). Use `if demands.size > 0:` or `if demands is not None:`.
    """
        return prompt

    @staticmethod
    def repair_initialization_prompt(task_description: str, existing_operators: list):
        operators_str = "\n".join([f"- {op['name']}: {op['code']}" for op in existing_operators])

        prompt = f"""
You are an expert in heuristic optimization algorithms, specifically Large Neighborhood Search (LNS).
Your task is to design a new 'Repair Operator' (insertion operator) for the following problem:

Problem Description:
{task_description}

Existing Repair Operators (Reference):
{operators_str}

    Requirement:
    1. First, describe your new algorithm and main steps in one sentence. The description must be inside a brace. Next, implement it in Python as a function named repair
    2. The logic should be innovative and distinct from the reference operators to ensure diverse reconstruction paths.
    3. Do not give additional explanations.
    4. IMPORTANT: Wrap the code in a Markdown code block (e.g., ```python ... ```).
    5. Do not assume any external functions exist other than standard Python libraries and what is provided in problem_data.
    6. Do NOT evaluate numpy arrays as booleans (e.g. `if demands:`). Use `if demands.size > 0:` or `if demands is not None:`.
    """
        return prompt

    @staticmethod
    def mutation_prompt(operator_code: str, strategy: str, operator_type: str, task_description: str = ""):
        m1_instruction = "Generate novel algorithmic mechanisms or formulas to replace existing logic components."
        m2_instruction = "Adjust current parameter settings (e.g., the degree of randomization or greedy thresholds) to optimize operator behavior."

        advice = ""
        if strategy == 'best':
            advice = f"This is a high-performing operator. Apply {m2_instruction}"
        elif strategy == 'worst':
            advice = f"This operator performs poorly. Apply {m1_instruction}"
        else:
            if random.random() < 0.5:
                advice = f"This is a selected operator. Strategy: {m1_instruction}"
            else:
                advice = f"This is a selected operator. Strategy: {m2_instruction}"

        prompt = f"""
You are an algorithm optimizer. We have a {operator_type} operator for LNS.

Problem Description:
{task_description}

Strategy: {advice}

Current Code:
```python
{operator_code}
```

    Requirement:
    1. Refine and improve this operator strictly following the strategy provided above.
    2. If you need helper functions, define them INSIDE the main function.
    3. Do not give additional explanations.
    4. IMPORTANT: Wrap the code in a Markdown code block (e.g., ```python ... ```).
    5. Do NOT evaluate numpy arrays as booleans (e.g. `if demands:`). Use `if demands.size > 0:` or `if demands is not None:`.
    """
        return prompt

    @staticmethod
    def homogeneous_crossover_prompt(parent1_code: str, parent2_code: str, operator_type: str, task_description: str = ""):
        prompt = f"""
You are an expert in heuristic optimization.
Your task is to create a NEW {operator_type} operator by combining the ideas/logic of two parent operators.

Problem Description:
{task_description}

Parent 1 Code:
```python
{parent1_code}
```

Parent 2 Code:
```python
{parent2_code}
```

Task:
Please create a new algorithm that has a similar form to Parent 2 and is inspired by Parent 1. The new algorithm should outperform both parents.

Firstly, list the common ideas in Parent 1 that may give good performances. 
Secondly, based on the common idea, describe the design idea of the new algorithm and its main steps in one sentence.
Next, implement it in Python.

    Requirement:
    1. The new operator MUST follow the standard LNS {operator_type} signature strictly.
    2. Define all helper functions INSIDE the main function.
    3. If you need helper functions, define them INSIDE the main function.
    4. Do not give additional explanations.
    5. IMPORTANT: Wrap the code in a Markdown code block (e.g., ```python ... ```).
    6. Do NOT evaluate numpy arrays as booleans (e.g. `if demands:`). Use `if demands.size > 0:` or `if demands is not None:`.
    """
        return prompt

    @staticmethod
    def synergistic_joint_crossover_prompt(destroy_code: str, repair_code: str, task_description: str = ""):
        prompt = f"""
You are an expert in heuristic optimization.
We are employing a "Synergistic Joint Crossover (Structural Coupling)" strategy to evolve LNS operators.

Problem Description:
{task_description}

We have selected a high-performing Destroy-Repair Pair based on their synergy:

Parent Destroy Operator:
```python
{destroy_code}
```

Parent Repair Operator:
```python
{repair_code}
```

Task:
Evolve this pair as a UNIFIED ENTITY to create a new Destroy-Repair pair. The goal is to address the inherent coupling between destroy and repair actions. Specifically, ensure that the generated Repair operator is specifically tailored to reconstruct the structural defects introduced by the generated Destroy operator, thereby maximizing their synergistic performance.

    Requirement:
    1. Design a NEW Destroy operator and a NEW Repair operator.
    2. The new Destroy operator should create specific structural defects.
    3. The new Repair operator must be designed to fix these specific defects efficiently.
    4. Both must follow standard LNS signatures strictly.
    5. Define all helper functions INSIDE the main functions.
    6. Return ONE code block containing BOTH the new Destroy operator and the new Repair operator.
    7. Do not give additional explanations.
    8. IMPORTANT: Wrap the code in a Markdown code block (e.g., ```python ... ```).
    9. Do NOT evaluate numpy arrays as booleans (e.g. `if demands:`). Use `if demands.size > 0:` or `if demands is not None:`.
    """
        return prompt
