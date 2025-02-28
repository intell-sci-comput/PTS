import os
import sys
import argparse
import warnings
import time
import numpy as np
from numpy import *
import pandas as pd
from sympy import simplify, expand
from contextlib import redirect_stdout
from dso import DeepSymbolicOptimizer

warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

os.environ["CUDA_VISIBLE_DEVICES"]="1"

data_folder = 'regression_task/data/'
config_folder = 'regression_task/configs_dsr/'
output_folder = 'regression_task/results_dsr/'

parser = argparse.ArgumentParser(description='dsr')
parser.add_argument(
    '--task',
    default='nguyen-1',
    type=str, help="""please select the benchmark task from the list 
                      [nguyen-1, nguyen-2, nguyen-3, nguyen-4, nguyen-5, nguyen-6, 
                       nguyen-7, nguyen-8, nguyen-9, nguyen-10, nguyen-11, nguyen-12, 
                       nguyen-1c, nguyen-2c, nguyen-5c, nguyen-8c, nguyen-9c]""")
parser.add_argument(
    '--num_test',
    default=100,
    type=int, help='number of tests performed, default 100')
parser.add_argument(
    '--norm_threshold',
    default=1e-4,
    type=float, help='the highest MSE for testing to assert a succesful discovery resultm defaul 1e-4')
parser.add_argument(
    '--save_solutions',
    default=True,
    type=bool, help='whether or not save discovered equations, default true')
args = parser.parse_args()


def simplify_eq(eq):
    return str(expand(simplify(eq)))


def main(args):
    task = args.task
    num_test = args.num_test
    norm_threshold = args.norm_threshold
    save_eqs = args.save_solutions

    all_times = []
    all_eqs = []
    num_success = 0

    ## define testing independent variables 
    test_sample = pd.read_csv(data_folder + task + '_test.csv', header=None).to_numpy()
    num_var = test_sample.shape[1] - 1
    current_var_test = 'x'
    for i in range(num_var):
        globals()[current_var_test] = test_sample[:, i]
        current_var_test = chr(ord(current_var_test) + 1)
    f_true = test_sample[:, -1]

    for i_test in range(num_test):
        print("\rTest {}/{}.".format(i_test, num_test), end="")
        sys.stdout.flush()
        
        start_time = time.time()
        
        with redirect_stdout(None):
            model = DeepSymbolicOptimizer(config_folder + task + ".json")
            result = model.train()
        dsr_eq = simplify_eq(result['expression'][1:-1].replace('x1', 'x').replace('x2', 'y').replace('x3', 'z'))
        
        end_time = time.time() - start_time
        all_eqs.append(dsr_eq)
        
        
        try: 
            f_pred = eval(dsr_eq)
            
            if np.linalg.norm(f_pred - f_true, 2) / f_true.shape[0] <= norm_threshold:
                num_success += 1
                all_times.append(end_time)
        except NameError:
            continue 

    if save_eqs:
        output_file = open(output_folder + task + '.txt', 'w')
        for eq in all_eqs:
            output_file.write(eq + '\n')
        output_file.close()

    print()
    print('final result:')
    print('success rate :', "{:.0%}".format(num_success / num_test))
    print('average discovery time is', np.round(np.mean(all_times), 3), 'seconds')


if __name__ == '__main__':
    main(parser.parse_args())