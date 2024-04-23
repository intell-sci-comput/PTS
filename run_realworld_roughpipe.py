# 23checked
import os
import time
import numpy as np
import sympy as sp
import torch

from model.regressor import PSRN_Regressor
from utils.data import get_dynamic_data

import click
@click.command()
@click.option('--gpu_index','-g',default=0,type=int,help='gpu index used')
@click.option('--n_runs','-r',default=20,type=int,help='number of runs for each puzzle')
def main(gpu_index, n_runs):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########### load data #############

    df, variables_name, target_name = get_dynamic_data('roughpipe','nikuradze')

    logf = df['y'].values.reshape(len(df),-1)
    logRe = df['l'].values.reshape(len(df),-1)
    invRelativeRoughness = df['k'].values.reshape(len(df),-1)

    f = 10 ** logf / 100
    Re = 10 ** logRe

    X = np.log10(Re*np.sqrt(f/32)*(1/invRelativeRoughness))
    Y = f ** (-1/2) + 2 * np.log10(1/invRelativeRoughness)

    Input = torch.from_numpy(X).reshape(len(df),-1)
    Output = torch.from_numpy(Y).reshape(len(df),-1)

    Input = Input.to(device).to(torch.float32)
    Output = Output.to(device).to(torch.float32)

    print(Input.shape, Output.shape) 
    print(Input.dtype, Output.dtype)

    ops = ['Add','Mul', 'Sub','SemiDiv','Identity','Pow2','Pow3','Cos','Cosh','Exp','Log','Sin','Tanh']

    down = 100
    simu = 5

    import gc
    for seed in range(n_runs):
        gc.collect()
        
        np.random.seed(seed)
        regressor = PSRN_Regressor(variables=['x'],
                                    operators=ops,
                                    n_symbol_layers=3,
                                    n_inputs=5,
                                    use_const=True,
                                    trying_const_num=3,
                                    trying_const_range=[0,3],
                                    trying_const_n_try=1,
                                    device='cuda',
                                    )

        start = time.time()
        flag, pareto_ls = regressor.fit( Input,
                                                Output,
                                                n_down_sample=down,
                                                n_step_simulation=simu,
                                                threshold=1e-6,
                                                probe=None,
                                                prun_const=True,
                                                prun_ndigit=3,
                                                real_time_display=True,
                                                real_time_display_freq=1,
                                                real_time_display_ntop=20,
                                                top_k=30,
                                                use_replace_expo=False,
                                                run_epochs=1,
                                            )
        end = time.time()
        time_cost = end - start


        crits = ['reward','mse']

        for crit in crits:
            print('Pareto Front sort by {}'.format(crit))
            pareto_ls = regressor.display_expr_table(sort_by=crit)

        from utils.log_ import save_pareto_frontier_to_csv, create_dir_if_not_exist

        p = './log/roughpipe/'
        fn = 'pf_{}.csv'.format(seed)
        create_dir_if_not_exist(p)
        with open(p+'time.txt','a') as f:
            f.write(str(time_cost)+'\n')
        save_pareto_frontier_to_csv(p + fn, pareto_ls)

if __name__ == '__main__':
    main()