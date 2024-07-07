import os
import click
import time
import numpy as np
import sympy as sp
import pandas as pd
import torch

import gc


def if_is_exist(path,name):
    for root, dirs, files in os.walk(path):
        for file in files:
            if name in file:
                return True
    return False



from model.regressor import PSRN_Regressor
from utils.data import get_benchmark_data, expr_to_Y_pred

from sklearn.metrics import r2_score

import traceback

@click.command()
@click.option('--experiment_name',default='_',type=str,help='experiment_name')
@click.option('--n_runs','-r',default=20,type=int,help='number of runs for each puzzle')
@click.option('--gpu_index','-g',default=0,type=int,help='gpu index used')
@click.option('--library','-l',default='koza',type=str,help='operator library')
@click.option('--n_down_sample','-d',default=40,type=int,help='n sample to downsample in PSRN for speeding up')
@click.option('--n_step_simulation','-p',default=400,type=int,help='number of MCTS simulations in each PTS epoch')
@click.option('--eta','-e',default=0.99,type=float,help='eta in reward equation')
@click.option('--n_inputs','-i',default=5,type=int,help='PSRN input size (n variables + n constants)')
@click.option('--seed','-s',default=0,type=int,help='seed')
@click.option('--benchmark_file','-b',default='benchmark.csv',type=str,help='benchmark csv name')
@click.option('--topk','-k',default=10,type=int,help='number of best expressions to take from PSRN to fit')
def main(experiment_name, n_runs, gpu_index, library, n_down_sample, n_step_simulation, eta, n_inputs, seed, benchmark_file, topk):
    '''

    >>> python run_benchmark_all.py --n_runs 100 -g 0 -l koza -i 5 -b benchmark.csv

    For the Feynman expressions:

    >>> python run_benchmark_all.py --n_runs 100 -g 0 -l semi_koza -i 6 -b benchmark_Feynman.csv

    '''

    os.environ['CUDA_VISIBLE_DEVICES']= str(gpu_index) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)
    random_seed = seed

    hp = {
        'n_runs':n_runs,
        'library':library,
        'n_down_sample':n_down_sample,
        'n_step_simulation':n_step_simulation,
        'eta':eta,
        'n_inputs':n_inputs,
        'topk':topk,
        'seed':seed,
        }

    for key, value in hp.items():
        experiment_name += '_{}'.format(value)


    ########### load data #############

    path_log = './log/benchmark/' + experiment_name + '/'

    if not os.path.exists(path_log):    
        os.makedirs(path_log)

    df = pd.read_csv('./benchmark/{}'.format(benchmark_file))

    df_save_all = pd.DataFrame(columns=['name', 'recovery_rate', 'avg_time_cost','n_runs'])


    err_txt_path = path_log + 'error_file.txt'

    
    

    for benchmark_name in df['name']:
        
        if if_is_exist(path_log[:-1],benchmark_name):
            continue
        
        if not os.path.exists(path_log[:-1]+'/'+benchmark_name):    
            print('making',path_log[:-1]+'/'+benchmark_name)
            os.makedirs(path_log[:-1]+'/'+benchmark_name)
        
        try:

            print('Runing benchmark: {}'.format(benchmark_name))

            cnt_success = 0
            sum_time = 0
            
            print('n_runs: {}'.format(n_runs))

            df_save = pd.DataFrame(columns=['name', 'success', 'time_cost',
                                             'R2', 'MSE', 'reward', 'complexity','expr_str_best_Reward',
                                        'expr_sympy_best_Reward','expr_str_best_MSE', 'expr_sympy_best_MSE'])

            for i in range(n_runs):
                
                gc.collect()

                np.random.seed(random_seed + i)
                
                print('Runing {}-th time'.format(i+1))
                
                lls = os.listdir(path_log + benchmark_name)
                is_continue = False
                for ll in lls:
                    print(ll)
                    if ll.startswith('hof_{}'.format(i)):
                        print('continue', i)
                        is_continue = True
                        break
                if is_continue:
                    continue
                
                Input, Output, use_constant, expression, variables_name = get_benchmark_data(benchmark_file,
                    benchmark_name)

                Input = torch.from_numpy(Input).to(device).to(torch.float32)
                Output = torch.from_numpy(Output).to(device).to(torch.float32)

                print(Input.shape, Output.shape)
                print(Input.dtype, Output.dtype)
                
                if hp['library'] == 'basic':
                    operators = ['Add', 'Mul',
                                'Identity', 'Neg','Inv', 'Sin', 'Cos', 'Exp', 'Log']
                elif hp['library'] == 'semi_koza':
                    operators = ['Add', 'Mul', 'SemiSub', 'SemiDiv',
                                'Identity', 'Neg','Inv', 'Sin', 'Cos', 'Exp', 'Log']
                elif hp['library'] == 'koza':
                    operators = ['Add', 'Mul', 'Sub', 'Div',
                                'Identity', 'Sin', 'Cos', 'Exp', 'Log']
                else:
                    raise ValueError('Unknown library: {}'.format(hp['library']))

                regressor = PSRN_Regressor(variables=variables_name,
                                        operators=operators,
                                        n_symbol_layers=3,
                                        n_inputs=hp['n_inputs'],
                                        dr_mask_dir='./dr_mask',
                                        use_const=use_constant,
                                        trying_const_num=2,
                                        trying_const_range=[0,3],
                                        trying_const_n_try=3,
                                        device='cuda',
                                        )

                start = time.time()
                flag, pareto_ls = regressor.fit(Input,
                                                    Output,
                                                    n_down_sample=hp['n_down_sample'],
                                                    n_step_simulation=hp['n_step_simulation'],
                                                    eta=hp['eta'],
                                                    use_threshold=False,   # Not use threshold when running benchmarks
                                                    threshold=1e-25,
                                                    probe=expression,
                                                    prun_const=True,
                                                    prun_ndigit=2,
                                                    real_time_display=True,
                                                    real_time_display_freq=1,
                                                    real_time_display_ntop=10,
                                                    dc=0.1,             # constant sampling interval
                                                    top_k=topk, 
                                                    )
                end = time.time()
                time_cost = end - start

                ############# Print Pareto Front ###############
                crits = ['mse', 'reward']
                
                expr_str_best_reward = None
                expr_sympy_best_reward = None
                expr_str_best_MSE = None
                expr_sympy_best_MSE = None

                for crit in crits:
                    print('Pareto Front sort by {}'.format(crit))
                    pareto_ls = regressor.display_expr_table(sort_by=crit)
                    expr_str, reward, loss, complexity = pareto_ls[0]
                    
                    if crit == 'mse':
                        expr_str_best_MSE = expr_str
                        expr_sympy_best_MSE = sp.simplify(expr_str)
                    else:
                        expr_str_best_reward = expr_str
                        expr_sympy_best_reward = sp.simplify(expr_str)  
                         
                print(expr_str)

                print('time_cost', time_cost)
                if flag:
                    print('[*** Found Expr ! ***]')
                    cnt_success += 1
                sum_time += time_cost

                print('----- expr_sympy_best_MSE -----')
                print(expr_sympy_best_MSE)
                print('----- expr_sympy_best_reward -----')
                print(expr_sympy_best_reward)
                
                ############### Pred ##################

                X_plot = Input.cpu().numpy()
                Y_pred_plot = expr_to_Y_pred(expr_sympy_best_reward, X_plot, variables_name)
                Y_plot = Output.cpu().numpy()

                ############## R2 #########################

                R2 = r2_score(Y_plot.ravel(), Y_pred_plot.ravel())
                print('R2', R2)

                ############## Plot #############################

                df_save = df_save.append({'name': benchmark_name, 'success': flag, 'time_cost': time_cost, 'expr_str_best_Reward': expr_str_best_reward,
                                        'expr_sympy_best_Reward': expr_sympy_best_reward,'expr_str_best_MSE': expr_str_best_MSE,
                                        'expr_sympy_best_MSE': expr_sympy_best_MSE, 'R2': R2, 'MSE': loss, 'reward': reward, 'complexity': complexity}, ignore_index=True)

                df_hof = pd.DataFrame(pareto_ls, columns=['expr_str', 'reward', 'MSE', 'complexity'])
                df_hof = df_hof.head(20)
                t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

                if not os.path.exists(path_log + '/{}'.format(benchmark_name)):
                    os.makedirs(path_log + '/{}'.format(benchmark_name))
                df_hof.to_csv(path_log + '{}/pf_{}_{}.csv'.format(benchmark_name, i, t), index=False)

        except Exception as e:
            traceback_info = traceback.format_exc()
            
            with open(err_txt_path, "a") as f:
                f.write(str(traceback_info))
            raise ValueError
        
        # save df_save
        t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        df_save.to_csv(path_log + 'benchmark_{}_{}.csv'.format(benchmark_name, t), index=False)

        avg_time = sum_time / n_runs
        avg_success_rate = cnt_success / n_runs
        df_save_all = df_save_all.append(
            {'name': benchmark_name, 'recovery_rate': avg_success_rate, 'avg_time_cost': avg_time, 'n_runs': n_runs}, ignore_index=True)

        print(df_save_all)

    # save df_save_all
    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    df_save_all.to_csv(path_log + 'benchmark_all_{}.csv'.format(t))

if __name__ == '__main__':
    main()