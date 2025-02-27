import os
import time
import numpy as np
import sympy as sp
from pysr import PySRRegressor
import pandas as pd
from utils.data import get_dynamic_data

df, variables_name, target_name = get_dynamic_data('roughpipe','nikuradze')

logf = df['y'].values.reshape(len(df),-1)
logRe = df['l'].values.reshape(len(df),-1)
invRelativeRoughness = df['k'].values.reshape(len(df),-1) 

f = 10 ** logf / 100
Re = 10 ** logRe

X = np.log10(Re*np.sqrt(f/32)*(1/invRelativeRoughness)) 
Y = f ** (-1/2) + 2 * np.log10(1/invRelativeRoughness) 

seed = 0
binary_ops = ['+','-','*','/']
unary_ops = ['cos','sin','exp','log','tanh','cosh','square','cube']

for seed in range(20):


    np.random.seed(seed)
    model = PySRRegressor(
        random_state=seed,
        deterministic=True,
        procs=0,
        multithreading=False,
        niterations=50,
        binary_operators=binary_ops,
        unary_operators=unary_ops,
        extra_sympy_mappings={
                            "square": lambda x: x**2,
                            "cube": lambda x: x**3}
    )
    Input = X
    Output = Y
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

    p = './log/roughpipe/PYSR/'
    with open(p+'time.txt','a') as f:
        f.write(str(time_cost)+'\n')
    df.to_csv(p+'pf_{}.csv'.format(seed))