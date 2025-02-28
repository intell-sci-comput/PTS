import sys
import argparse
import pandas as pd
import numpy as np
from numpy import *
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

sys.path.append(r'../')
from spl_train import run_spl

data_folder = 'data/'          ## directory to data folder
output_folder = 'results_spl/' ## directory to save results
image_folder = 'pictures/'     ## directory to save images

font = font_manager.FontProperties(family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=44)


baseline_models = ['C+C*x+C*x**2+C*x**3', 
                   'C+C*x+C*exp(C*x)', 
                   'C+C*log(cosh(C*x))']


parser = argparse.ArgumentParser(description='dsr')
parser.add_argument(
    '--task',
    default='Baseball',
    type=str, help="""please select the benchmark task from the list 
                       ['Baseball',
                        'Blue_Basketball',
                        'Green_Basketball',
                        'Volleyball',
                        'Bowling_Ball',
                        'Golf_Ball',
                        'Tennis_Ball',
                        'Whiffle_Ball_1',
                        'Whiffle_Ball_2',
                        'Yellow_Whiffle_Ball',
                        'Orange_Whiffle_Ball']]""")
parser.add_argument(
    '--save_solutions',
    default=True,
    type=bool, help='whether or not save discovered equations, default true')
args = parser.parse_args()


def solve_model(eq, data):
    x = data[0, :]
    f_true = data[1, :]
    c_count = eq.count('C')
    c_lst = ['c'+str(i) for i in range(c_count)]
    for c in c_lst: 
        eq = eq.replace('C', c, 1)

    def eq_test(c):
        x = data[0, :]
        for i in range(len(c)): 
            globals()['c'+str(i)] = c[i]
        return np.linalg.norm(eval(eq) - f_true, 2)
    
    x0 = [1.0] * len(c_lst)
    c_lst = minimize(eq_test, x0, method='Powell', tol=1e-6).x.tolist()
    c_lst = [np.round(x, 3) if abs(x) > 1e-2 else 0 for x in c_lst]
    eq_est = eq
    for i in range(len(c_lst)):
        eq_est = eq_est.replace('c'+str(i), str(c_lst[i]), 1)
    return eq_est.replace('+-', '-')


def pred_with_baseline_model(model, train_sample, test_sample):
    eq_model = solve_model(model, train_sample)
    x = test_sample[0, :]
    f_pred = eval(eq_model)
    return f_pred, eq_model.replace('x', 't').replace('etp', 'exp')


def main(args):
    task = args.task.replace('_', ' ')
    save_solutions = args.save_solutions

    train_sample = pd.read_csv(data_folder + task + '_train.csv', header=None).to_numpy().T
    test_sample = pd.read_csv(data_folder + task + '_test.csv', header=None).to_numpy().T

    f_model1, eq_model1 = pred_with_baseline_model(baseline_models[0], train_sample, test_sample)
    f_model2, eq_model2 = pred_with_baseline_model(baseline_models[1], train_sample, test_sample)
    f_model3, eq_model3 = pred_with_baseline_model(baseline_models[2], train_sample, test_sample)
    
    all_eqs, _, _ = run_spl(task, 
                            num_run=1, 
                            max_len=20,
                            eta=0.9999, 
                            num_transplant=3, 
                            num_aug=0,
                            transplant_step=2000, 
                            count_success=False)
    
    x = test_sample[0, :]
    f_true = test_sample[1, :]
    f_srl = eval(all_eqs[0])
    eq_spl = all_eqs[0].replace('x', 't').replace('etp', 'exp')
    
    print('SPL model  :', eq_spl, '  MSE :', np.round(mean_squared_error(f_true,f_srl), 3))
    print('Model-1  :', eq_model1, '  MSE :', np.round(mean_squared_error(f_true,f_model1), 3))
    print('Model-2  :', eq_model2, '  MSE :', np.round(mean_squared_error(f_true,f_model2), 3))
    print('Model-3  :', eq_model3, '  MSE :', np.round(mean_squared_error(f_true,f_model3), 3))

    if save_solutions: 
        output_file = open(output_folder + task + '.txt', 'w')
        for eq in [eq_spl, eq_model1, eq_model2, eq_model3]:
            output_file.write(eq + '\n')
        output_file.close()

    
        fig = plt.figure(figsize=(9, 12))
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.scatter(x, f_true, s=100, color='black', label="True measurement")
        ax.plot(x, f_srl, '-.', c='tomato', lw=5, label="SPL discovered")
        ax.plot(x, f_model1, ':', c='green', lw=5, label="Model-1 discovered")
        ax.plot(x, f_model2, '-', c='skyblue', lw=5, label="Model-2 discovered")
        ax.plot(x, f_model3, '-', c='brown', lw=5, label="Model-3 discovered")
        ax.set_xticks(np.arange(2, 3.001, 0.5))
        ax.set_xticklabels(np.arange(2, 3.001, 0.5))
        plt.legend(loc=[1.08, 0.05], frameon=False, fontsize=30, prop=font)
        ax.tick_params(axis='both', which='major', labelsize=20)

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        fig.savefig(image_folder + task + '.svg', bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    main(parser.parse_args())