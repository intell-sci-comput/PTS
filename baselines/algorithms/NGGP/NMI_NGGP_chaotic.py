import os 
import numpy as np
import pandas as pd

from dysts.flows import *
from pysindy import SmoothedFiniteDifference

from utils.data import add_noise
import gc

class_name_ls = open('../../../data/dystsnames/dysts_16_flows.txt').readlines()
class_name_ls = [line.strip() for line in class_name_ls]

print(class_name_ls)

sample_size = 1000
noise_level = 0.01
pts_per_period = 100
n_seeds = 50

for seed in range(n_seeds):

    experi_name = 'NGGP_chaotic'

    for class_name in class_name_ls:
            
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
        
        gc.collect()
        np.random.seed(seed)

        print(dysts_model)
        t, data = dysts_model.make_trajectory(sample_size,
                                                return_times=True,
                                                pts_per_period=pts_per_period,
                                                resample=True,
                                                noise=0,
                                                postprocess=False)
        
        # add Gaussian noise
        for k in range(data.shape[1]):
            data[:,k:k+1] = add_noise(data[:,k:k+1], noise_level, seed)
            
        t = t.flatten()
        # t: Nx1 ; sol:NxD
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
        data_all = np.hstack([data, data_deriv])
        
        binary_ops = ['+','-','*','/']
        unary_ops = ['sin','cos','exp','log','tanh','cosh','abs','sign']
        
        for idxdot, vardotname in enumerate(dotsvarnames):

            hp_str = '{}/{}/{}'.format(experi_name, class_name, vardotname)
            p = './log/dysts_{}/nggp/{}/'.format(noise_level,hp_str)
                
            import os 
            if os.path.exists(p+'pf_{}.csv'.format(seed)):
                print('exist {}, skip.'.format(p+'pf_{}.csv'.format(seed)))
                continue
            
            Input = data
            Output = data_deriv[:,idxdot:idxdot+1]
        
            benchmark_csv_filename = './dysts_nggp_{}_{}_{}.csv'.format(class_name, vardotname, seed)
            
            df = pd.DataFrame(np.hstack([Input, Output]),columns = vars + [vardotname])

            df.to_csv(benchmark_csv_filename, header=None,index=False)

            os.system("python -m dso.run ./json/NGGP_const_fast_dysts.json --b={} --runs=1 --n_cores_task={} --seed={}".format(
                            benchmark_csv_filename,
                            1,
                            seed
                        ))
