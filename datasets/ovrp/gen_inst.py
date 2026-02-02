import os
import numpy as np

CAPACITY = 50
DEMAND_LOW = 1
DEMAND_HIGH = 9
DEPOT_COOR = [0.5, 0.5]

def gen_instance(n):
    locations = np.random.rand(n, 2)
    demands = np.random.randint(low=DEMAND_LOW, high=DEMAND_HIGH+1, size=n)
    depot = np.array([DEPOT_COOR])
    all_locations = np.concatenate((depot, locations), axis=0)
    all_demands = np.concatenate((np.zeros(1,), demands))
    return np.concatenate((all_demands.reshape(-1, 1), all_locations), axis=1)

def generate_datasets():
    basepath = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(basepath, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    np.random.seed(1234)
    
    # Training datasets
    for problem_size in [20, 50]:
        n_instances = 16
        dataset = []
        for i in range(n_instances):
            inst = gen_instance(problem_size)
            dataset.append(inst)
        dataset = np.array(dataset)
        np.save(os.path.join(dataset_dir, f'train{problem_size}_dataset.npy'), dataset)
        print(f"Generated train{problem_size}_dataset.npy with {n_instances} instances.")

    # Validation datasets
    for problem_size in [20, 50, 100]:
        n_instances = 16
        dataset = []
        for i in range(n_instances):
            inst = gen_instance(problem_size)
            dataset.append(inst)
        dataset = np.array(dataset)
        np.save(os.path.join(dataset_dir, f'val{problem_size}_dataset.npy'), dataset)
        print(f"Generated val{problem_size}_dataset.npy with {n_instances} instances.")
    
    # Test datasets
    for problem_size in [10, 20, 50, 100, 200]:
        n_instances = 64
        dataset = []
        for i in range(n_instances):
            inst = gen_instance(problem_size)
            dataset.append(inst)
        dataset = np.array(dataset)
        np.save(os.path.join(dataset_dir, f'test{problem_size}_dataset.npy'), dataset)
        print(f"Generated test{problem_size}_dataset.npy with {n_instances} instances.")

if __name__ == "__main__":
    generate_datasets()
