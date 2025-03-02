#check
import os 
import gc
import torch
import numpy as np
import time
import pandas as pd
from dysts.flows import *
from pysindy import SmoothedFiniteDifference
from utils.data import add_noise, get_dynamic_data, get_benchmark_data
from utils.log_ import create_dir_if_not_exist, save_pareto_frontier_to_csv
from model.regressor import PSRN_Regressor
import click


@click.command()
@click.option('--gpu_index','-g',default=0,type=int,help='gpu index used')
@click.option('--n_runs','-r',default=50,type=int,help='number of runs for each puzzle')
def main(gpu_index, n_runs):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)

    class_name_ls = open('./data/dystsnames/dysts_16_flows.txt').readlines()
    class_name_ls = [line.strip() for line in class_name_ls]

    print(class_name_ls)

    # Number of random seeds for each chaotic dynamic
    n_seeds = n_runs

    # Chaotic dynamic data generation
    sample_size = 1000
    pts_per_period = 100
    noise_level = 0.01
    
    N_SIMU = 3
    postprocess = False 

    for seed in range(n_seeds):
        for class_name in class_name_ls:
            # Convert to autonomous system
            dysts_model = eval(class_name)
            dysts_model = dysts_model()
            if class_name == 'ForcedBrusselator':
                dysts_model.f = 0
            elif class_name == 'Duffing':
                dysts_model.gamma = 0
            elif class_name == 'ForcedVanDerPol':
                dysts_model.a = 0
            elif class_name == 'ForcedFitzHughNagumo':
                dysts_model.f = 0
            
            # clear PSRN's VRAM memory
            gc.collect()
            np.random.seed(seed) # setting random seed
            try:
                print('=== START SEARCHING FOR GT MODEL: ===')
                print(dysts_model) 
                # Gen traj data using `dysts` library
                t, data = dysts_model.make_trajectory(sample_size,
                                                        return_times=True,
                                                        pts_per_period=pts_per_period,
                                                        resample=True,
                                                        noise=0,
                                                        postprocess=postprocess)
                
                # introduce gaussian noise
                for k in range(data.shape[1]):
                    data[:,k:k+1] = add_noise(data[:,k:k+1], noise_level, seed)

                # Make derivation for each variable
                t = t.flatten()
                dim = data.shape[1]
                print('dim',dim)
                if dim == 3:
                    vars = ['x','y','z']
                    dotsvarnames = ['xdot','ydot','zdot']
                    deriv_idxs = [0,1,2]
                    trying_const_num = 2
                    n_inputs = 5
                elif dim == 4:
                    vars = ['x','y','z','w']
                    dotsvarnames = ['xdot','ydot','zdot','wdot']
                    deriv_idxs = [0,1,2,3]
                    trying_const_num = 1
                    n_inputs = 5
                else:
                    continue
                
                sfd = SmoothedFiniteDifference(smoother_kws={'window_length': 5})
                data_deriv = np.zeros((data.shape[0],len(deriv_idxs)))
                for i,idx in enumerate(deriv_idxs):
                    print(data[:,idx:idx+1].shape,t.shape)
                    deriv_data_i = sfd._differentiate(data[:,idx:idx+1], t)
                    data_deriv[:,i:i+1] = deriv_data_i

                for idxdot, vardotname in enumerate(dotsvarnames):
                    merged_data = np.hstack((data, data_deriv[:,idxdot:idxdot+1]))
                    np.savetxt(f'data/transformed/{class_name}_{vardotname}_{seed}.csv', merged_data, delimiter=',', fmt='%.6f')
                    
            except np.linalg.LinAlgError:
                print('np.linalg.LinAlgError (Intel bug)')
                continue
            
        df, variables_name, target_name = get_dynamic_data('emps','emps')
        
        # select the first half of the data as train set
        df = df.iloc[:len(df)//2,:]
        
        gc.collect()
        np.random.seed(seed)

        Input = df[variables_name].values
        Output = df[target_name].values.reshape(-1,1)
        
        merged_data = np.hstack((Input, Output))
        np.savetxt(f'data/transformed/emps_{seed}.csv', merged_data, delimiter=',', fmt='%.6f')
                    
    
        df, variables_name, target_name = get_dynamic_data('roughpipe','nikuradze')

        logf = df['y'].values.reshape(len(df),-1)
        logRe = df['l'].values.reshape(len(df),-1)
        invRelativeRoughness = df['k'].values.reshape(len(df),-1)

        f = 10 ** logf / 100
        Re = 10 ** logRe

        X = np.log10(Re*np.sqrt(f/32)*(1/invRelativeRoughness))
        Y = f ** (-1/2) + 2 * np.log10(1/invRelativeRoughness)

        merged_data = np.hstack((X,Y))
        np.savetxt(f'data/transformed/roughpipe_{seed}.csv', merged_data, delimiter=',', fmt='%.6f')
        
        for benchmark_file in ['benchmark.csv', "benchmark_Feynman.csv"]:
            df = pd.read_csv('./benchmark/{}'.format(benchmark_file))
            for benchmark_name in df['name']:
                X, Y, *_ = get_benchmark_data(benchmark_file, benchmark_name)
                merged_data = np.hstack((X,Y))
                
                np.savetxt(f'data/transformed/basicbenchmark_{benchmark_name}_{seed}.csv', merged_data, delimiter=',', fmt='%.6f')
        


if __name__ == '__main__':
    main()