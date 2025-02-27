import os
import time
import numpy as np
import pandas as pd
import sympy as sp
from pysr import PySRRegressor
from utils.data import get_benchmark_data
def if_is_exist(path,name):
    for root, dirs, files in os.walk(path):
        for file in files:
            if name in file:
                return True
    return False

random_seed = 0
n_runs = 20
pysr_n_iter = 10000
seed = random_seed
benchmark_file_ls = ['benchmark.csv','benchmark_Feynman.csv']

hp = {
    'n_runs':n_runs,
    'pysr_n_iter':pysr_n_iter,
    'seed':seed,
    }

experiment_name = 'PySR_bench'

for key, value in hp.items():
    experiment_name += '_{}'.format(value)

path_log = './log/' + experiment_name + '/'

for benchmark_file in benchmark_file_ls:

    df = pd.read_csv('../../../benchmark/{}'.format(benchmark_file))

    df_save_all = pd.DataFrame(columns=['name', 'recovery_rate', 'avg_time_cost','n_runs'])

    for benchmark_name in df['name']:

        print('Runing benchmark: {}'.format(benchmark_name))

        cnt_success = 0
        sum_time = 0
        
        print('n_runs: {}'.format(n_runs))

        df_save = pd.DataFrame(columns=['name', 'success', 'time_cost',
                                        'expr_str', 'expr_sympy', 'R2', 'MSE', 'reward', 'complexity'])

        for i in range(n_runs):
            np.random.seed(random_seed + i)
            print('Runing {}-th time'.format(i+1))
            
            log_path = './log/pysr/benchmark/{}/{}/'.format(experiment_name,benchmark_name)
            os.makedirs(log_path, exist_ok=True)
            csv_name = '{}.csv'.format(random_seed + i)
            log_path_csv = log_path + csv_name
            
            if if_is_exist(log_path, csv_name):
                continue
            
            X, Y, use_constant, expression, variables_name = get_benchmark_data(benchmark_file,
                                    benchmark_name,
                                    1000)

            if use_constant:
                pysr_const_weight = 1
            else:
                pysr_const_weight = 100

            Input = X
            Output = Y

            np.random.seed(random_seed + i)
            model = PySRRegressor(
                random_state=random_seed + i,
                early_stop_condition="f(loss, complexity) = (loss < 1e-8)",
                timeout_in_seconds=3600*3,
                deterministic=True,
                procs=0,
                multithreading=False,
                equation_file=log_path_csv,
                temp_equation_file=False,
                niterations=pysr_n_iter, 
                binary_operators=["+", "*", "-", "/"],
                complexity_of_constants=pysr_const_weight,
                unary_operators=[
                    "cos",
                    "exp",
                    "log",
                    "sin",
                ],
            )

            start_time = time.time()
            np.random.seed(0)
            model.fit(Input, Output)
            end_time = time.time()
            time_cost = end_time - start_time
            print('time_cost',time_cost)
            with open(log_path+'time.txt','a') as f:
                f.write(str(time_cost)+'\n')
            print(model)