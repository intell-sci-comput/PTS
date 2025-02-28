import argparse
import numpy as np
from numpy import *
import pandas as pd
from os.path import exists
from task_info import *

parser = argparse.ArgumentParser(description='make_data')
parser.add_argument(
    '--task',
    default='nguyen-1',
    type=str, help="""please select the benchmark task from the list 
                      [nguyen-1, nguyen-2, nguyen-3, nguyen-4, nguyen-5, nguyen-6, 
                       nguyen-7, nguyen-8, nguyen-9, nguyen-10, nguyen-11, nguyen-12]""")
parser.add_argument(
    '--num_train',
    default=1000,
    type=int, help='number of training points, default 1000')
parser.add_argument(
    '--num_test',
    default=1001,
    type=int, help='number of testing points, default 1001')
parser.add_argument(
    '--noise',
    default=0.00,
    type=float, help='percentage of random noise on training data, default 0.0')
parser.add_argument(
    '--seed',
    default=0,
    type=int, help='random seed for data sampling, default 0')
args = parser.parse_args()


def main(args):

    task = args.task
    num_train = args.num_train
    num_test = args.num_test
    n_ratio = args.noise
    seed = args.seed
    
    train_l, train_u, test_l, test_u = bounds[task]
    num_var = num_vars[task]

    ## define training independent variables 
    current_var_train = 'x'
    current_seed_train = seed
    var_train = []
    for i in range(num_var):
        np.random.seed(current_seed_train)
        globals()[current_var_train] = np.random.uniform(train_l, train_u, num_train)
        var_train.append(globals()[current_var_train])
        current_var_train = chr(ord(current_var_train) + 1)
        current_seed_train += 1
    
    ## get training dependent variable f_true
    function = task_true_eqs[task]
    f_true = eval(function)

    ## add random noise to f_true
    np.random.seed(seed)
    f_n = np.random.normal(0, 1, num_train)
    f_n = f_n / np.std(f_n)
    f_train = f_true + n_ratio * np.sqrt(np.mean(f_true**2)) * f_n
    train_sample = np.vstack(var_train + [f_train])

    ## define testing independent variables 
    current_var_test = 'x'
    var_test = []
    for i in range(num_var):
        globals()[current_var_test] = np.linspace(test_l, test_u, num_test)
        var_test.append(globals()[current_var_test])
        current_var_test = chr(ord(current_var_test) + 1)
    
    ## get testing dependent variable f_test
    f_test = eval(function)
    test_sample = np.vstack(var_test + [f_test])

    ## check if files exist and save data
    if exists('data/' + task + '_train.csv'): 
        while True:
            var = input(task + "_train.csv already exist. Do you want to overwrite it? [y]/[n]")
            if var == 'y':
                pd.DataFrame(train_sample.T).to_csv('data/' + task + '_train.csv', index=False, header=False)
                break 
            elif var == 'n': break
    else: pd.DataFrame(train_sample.T).to_csv('data/' + task + '_train.csv', index=False, header=False)

    if exists('data/' + task + '_test.csv'): 
        while True:
            var = input(task + "_test.csv already exist. Do you want to overwrite it? [y]/[n]")
            if var == 'y':
                pd.DataFrame(test_sample.T).to_csv('data/' + task + '_test.csv', index=False, header=False)
                break 
            elif var == 'n': break
    else: pd.DataFrame(test_sample.T).to_csv('data/' + task + '_test.csv', index=False, header=False)



if __name__ == '__main__':
    main(parser.parse_args())