import sys
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(cur_dir))
sys.path.append(root_dir)

from task.tsp.tsp_task import TSPTask
from tools.llm.llm_api_https import HttpsApi
from tools.profiler import ProfilerBase
from method.glns import GLNS
import dotenv
import os

dotenv.load_dotenv()

def main():
    llm = HttpsApi(host=os.getenv('DEESEEK_API_HOST'),
                   key=os.getenv('DEESEEK_API_KEY'),
                   model=os.getenv('DEESEEK_API_MODEL'),
                   timeout=180)

    tspTask = TSPTask(n_instance=16, n_cities=50, seed=1234)

    alns_kwargs = {
        'T_start': 100,
        'alpha': 0.97,
        'max_iter': 100,
        'lambda_rate': 0.5,
        'reward_sigma1': 1.5,
        'reward_sigma2': 1.2,
        'reward_sigma3': 0.8,
        'reward_sigma4': 0.1,
    }

    method = GLNS(llm=llm,
                   task=tspTask,
                   profiler=ProfilerBase(log_dir='./logs/glns_tsp', log_style='complex'),
                   max_generations=400,
                   pop_size=5,
                   prune_size=2,
                   debug_mode=True,
                   **alns_kwargs)
    
    method.run()

if __name__ == "__main__":
    main()