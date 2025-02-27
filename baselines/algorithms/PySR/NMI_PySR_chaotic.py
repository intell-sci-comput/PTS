import os 
import gc
import time
import numpy as np
import pandas as pd
from dysts.flows import *
from pysindy import SmoothedFiniteDifference
from pysr import PySRRegressor
from utils.data import add_noise

class_name_ls = open('../../../data/dystsnames/dysts_16_flows.txt').readlines()
class_name_ls = [line.strip() for line in class_name_ls]

print(class_name_ls)

sample_size = 1000
noise_level = 0.01
n_iter = 300
pts_per_period = 100
n_seeds = 50

for seed in range(50):
    
    experi_name = 'chaocic'
    
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
            p = './log/dysts/pysr/{}/'.format(hp_str)
                
            
            if os.path.exists(p+'pf_{}.csv'.format(seed)):
                print('exist {}, skip.'.format(p+'pf_{}.csv'.format(seed)))
                continue
            
            Input = data
            Output = data_deriv[:,idxdot:idxdot+1]
        
            np.random.seed(seed)
            model = PySRRegressor(
                timeout_in_seconds=60*10,
                random_state=seed,
                deterministic=True,
                procs=0,
                multithreading=False,
                niterations=n_iter,
                binary_operators=binary_ops,
                unary_operators=unary_ops
            )

            start_time = time.time()
            np.random.seed(seed)
            model.fit(Input, Output)
            end_time = time.time()
            time_cost = end_time - start_time
            print('time_cost',time_cost)
            
            print(model)

            n = len(model.equations_)
            print(model.sympy([i for i in range(n)]))

            se = model.sympy([i for i in range(n)])
            df = pd.DataFrame(se)
            
            os.makedirs(p, exist_ok=True)
            with open(p+'time.txt','a') as f:
                f.write(str(time_cost)+'\n')
            df.to_csv(p+'pf_{}.csv'.format(seed))

