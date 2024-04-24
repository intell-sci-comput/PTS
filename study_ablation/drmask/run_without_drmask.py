import sys
sys.path.append('.')

import os
import click
import time
import numpy as np
import torch

from model.regressor import PSRN_Regressor

@click.command()
@click.option('--use_drmask','-u',default=False,type=bool,help='whether to use drmask')
@click.option('--n_inputs','-i',default=4,type=int,help='number of input slots of PSRN')
@click.option('--library','-l',default='koza',type=str,help='operator library')
@click.option('--gpu_idx','-g',default=0,type=int,help='whether to use drmask')
def main(use_drmask, n_inputs, library, gpu_idx):

    '''
    >>> python study_ablation/drmask/run_without_drmask.py --use_drmask True -i 4 -l koza -g 0
    
    >>> python study_ablation/drmask/run_without_drmask.py --use_drmask False -i 4 -l koza -g 0
    
    ########### VRAM usages: ##########
    # 30.39GB 4 koza False
    # 12.32GB 4 koza True
    # 35.50GB 5 semi_koza False
    # 17.05GB 5 semi_koza True
    # [OOM]   6 semi_koza False
    # 50.56GB 6 semi_koza True
    '''

    os.environ['CUDA_VISIBLE_DEVICES']= str(gpu_idx) # set

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Input = np.random.rand(100,1)
    Output = np.random.rand(100,1)

    Input = torch.from_numpy(Input).to(device).to(torch.float32)
    Output = torch.from_numpy(Output).to(device).to(torch.float32)

    print(Input.shape, Output.shape)
    print(Input.dtype, Output.dtype)

    hp = {
        'use_dr_mask':use_drmask,
        'library':library,
        'n_inputs':n_inputs,
        'seed':0,
        }

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

    regressor = PSRN_Regressor(variables=['x'],
                            operators=operators,
                            n_symbol_layers=3,
                            n_inputs=hp['n_inputs'],
                            use_dr_mask=hp['use_dr_mask'],
                            dr_mask_dir='./dr_mask',
                            use_const=False,
                            device='cuda',
                            )

    start = time.time()
    flag, pareto_ls = regressor.fit(Input,
                                        Output,
                                        n_down_sample=100,
                                        n_step_simulation=400,
                                        eta=0.99,
                                        use_threshold=False,   # Not use threshold when running benchmarks
                                        threshold=1e-25,
                                        probe=None,
                                        prun_const=True,
                                        prun_ndigit=2,
                                        real_time_display=True,
                                        real_time_display_freq=20,
                                        real_time_display_ntop=5,
                                        )
    end = time.time()
    time_cost = end - start

if __name__ == '__main__':
    main()