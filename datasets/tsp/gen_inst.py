import os
import numpy as np

def gen_instance(n):
    return np.random.rand(n, 2)

def generate_datasets():
    basepath = os.path.dirname(__file__)
    os.makedirs(os.path.join(basepath, "dataset"), exist_ok=True)
    
    np.random.seed(1234)
    
    for problem_size in [50]:
        n_instances = 10
        dataset = []
        for i in range(n_instances):
            inst = gen_instance(problem_size)
            dataset.append(inst)
        dataset = np.array(dataset)
        np.save(os.path.join(basepath, f'dataset/train{problem_size}_dataset.npy'), dataset)

    for problem_size in [20, 50, 100]:
        n_instances = 64
        dataset = []
        for i in range(n_instances):
            inst = gen_instance(problem_size)
            dataset.append(inst)
        dataset = np.array(dataset)
        np.save(os.path.join(basepath, f'dataset/val{problem_size}_dataset.npy'), dataset)
    
    for problem_size in [10, 20, 50, 100, 200, 500, 1000]:
        n_instances = 64
        dataset = []
        for i in range(n_instances):
            inst = gen_instance(problem_size)
            dataset.append(inst)
        dataset = np.array(dataset)
        np.save(os.path.join(basepath, f'dataset/test{problem_size}_dataset.npy'), dataset)

if __name__ == "__main__":
    generate_datasets()
