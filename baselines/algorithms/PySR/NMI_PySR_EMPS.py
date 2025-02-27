import os 
import time
import numpy as np
import pandas as pd
from pysr import PySRRegressor
from utils.data import get_dynamic_data

n_seeds = 20
n_iter = 100
TIMEOUT = 90
for seed in range(n_seeds):
    df, variables_name, target_name = get_dynamic_data('emps','emps')
    # select half of the data as train set
    df = df.iloc[:len(df)//2,:]
    
    np.random.seed(seed)
    
    Input = df[variables_name].values
    Output = df[target_name].values.reshape(-1,1)

    binary_ops = ['+','-','*','/']
    unary_ops = ['sin','cos','exp','log','sign','abs','cosh','tanh']
            
    p =  './log/EMPS/'
    
    if os.path.exists(p+'pf_{}.csv'.format(seed)):
        print('exist {}, skip.'.format(p+'pf_{}.csv').format(seed))
        continue

    np.random.seed(seed)
    model = PySRRegressor(
        timeout_in_seconds=TIMEOUT,
        random_state=seed,
        deterministic=True,
        procs=0,
        multithreading=False,
        niterations=n_iter,
        binary_operators=binary_ops,
        unary_operators=unary_ops,
        extra_sympy_mappings={"inv": lambda x: 1 / x,
                                "neg": lambda x: -x},
    )
    Input = df[variables_name].values
    Output = df[target_name].values.reshape(-1,1)
    
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
