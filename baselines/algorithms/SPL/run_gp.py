import os
import sys
import argparse
import warnings
import time
import numpy as np
from numpy import *
import pandas as pd
from contextlib import redirect_stdout
from gplearn.genetic import SymbolicRegressor

from spl_task_utils import f_set

warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

data_folder = 'regression_task/data/'
output_folder = 'regression_task/results_gp/'

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
    '--population',
    default=2000,
    type=int, help='number of initial populations in GP, default 2000')
parser.add_argument(
    '--generation',
    default=20,
    type=int, help='number of generations in GP, default 20')
parser.add_argument(
    '--const',
    default=False,
    type=bool, help='if there is anny constant to be estimated')
parser.add_argument(
    '--norm_threshold',
    default=1e-4,
    type=float, help='the highest MSE for testing to assert a succesful discovery resultm defaul 1e-4')
parser.add_argument(
    '--save_solutions',
    default=True,
    type=bool, help='whether or not save discovered equations, default true')
args = parser.parse_args()


variables = ['x', 'y', 'z']


def main(args):
    task = args.task
    num_test = args.num_test
    population = args.population
    generation = args.generation
    norm_threshold = args.norm_threshold
    save_eqs = args.save_solutions
    if args.const:
        range_const = (-10, 10)
    else:
        range_const = None

    all_times = []
    all_eqs = []
    num_success = 0
    
    ## define training variables 
    train_sample = pd.read_csv(data_folder + task + '_train.csv', header=None).to_numpy()
    x_train = train_sample[:, :-1]
    f_train = train_sample[:, -1]
    
    ## define testing variables 
    test_sample = pd.read_csv(data_folder + task + '_test.csv', header=None).to_numpy()
    x_test = test_sample[:, :-1]
    f_test = test_sample[:, -1]

    for i_test in range(num_test):
        print("\rTest {}/{}.".format(i_test, num_test), end="")
        sys.stdout.flush()
        
        start_time = time.time()
        
        # with redirect_stdout(None):
        est_gp = SymbolicRegressor(population_size=population, generations=generation, 
                                    stopping_criteria=0.01,const_range=range_const,
                                    max_samples=1, function_set=f_set[task], 
                                    feature_names=variables[:x_train.shape[1]])

        est_gp.fit(x_train, f_train)
        gp_eq = str(est_gp._program)
        
        end_time = time.time() - start_time
        all_eqs.append(gp_eq)
        
        
        try: 
            f_pred = est_gp.predict(x_test)

            print(np.linalg.norm(f_pred - f_test, 2) / f_test.shape[0])
            
            if np.linalg.norm(f_pred - f_test, 2) / f_test.shape[0] <= norm_threshold:
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