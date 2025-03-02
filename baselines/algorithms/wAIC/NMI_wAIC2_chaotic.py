import os 
import gc
import numpy as np
import time
import pandas as pd
from dysts.flows import *
from pysindy import SmoothedFiniteDifference
from utils.data import add_noise
from utils.log_ import create_dir_if_not_exist, save_pareto_frontier_to_csv
from waic.TwoPhaseInference import TwoPhaseInference
import click

def construct_equation(InferredResults, variables):
    terms = []
    for col in InferredResults.columns:
        coeff_series = InferredResults[col]
        if isinstance(coeff_series, pd.Series):
            coeff = coeff_series.mean()
        else:
            coeff = coeff_series
        
        if abs(coeff) > 1e-6:  # Only include non-zero terms
            if col in variables:
                terms.append(f"{coeff:.4f}*{col}")
            else:
                terms.append(f"{coeff:.4f}*{col}")
    return " + ".join(terms) if terms else "0"

def parse_inferred_results(InferredResults):
    terms = []
    for index, row in InferredResults.iterrows():
        if index == 0:  # Skip the first row if it's all zeros
            continue
        coeff = row.mean()
        if abs(coeff) > 1e-6:  # Only include non-zero terms
            term = f"{coeff:.6f}*{index}" if index != '1' else f"{coeff:.6f}"
            terms.append(term)
    
    equation = " + ".join(terms) if terms else "0"
    return equation

@click.command()
@click.option('--n_runs','-r',default=50,type=int,help='number of runs')
def main(n_runs):
    
    class_name_ls = open('../../../data/dystsnames/dysts_16_flows.txt').readlines()
    class_name_ls = [line.strip() for line in class_name_ls]

    print(class_name_ls)

    n_seeds = n_runs

    sample_size = 1000
    pts_per_period = 100
    noise_level = 0.01

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
            
            np.random.seed(seed) # setting random seed

            print('=== START SEARCHING FOR GT MODEL: ===')
            print(dysts_model) 
            
            t, data = dysts_model.make_trajectory(sample_size,
                                                    return_times=True,
                                                    pts_per_period=pts_per_period,
                                                    resample=True,
                                                    noise=0,
                                                    postprocess=False)
            
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
            elif dim == 4:
                vars = ['x','y','z','w']
                dotsvarnames = ['xdot','ydot','zdot','wdot']
                deriv_idxs = [0,1,2,3]
            else:
                continue
            sfd = SmoothedFiniteDifference(smoother_kws={'window_length': 5})
            data_deriv = np.zeros((data.shape[0],len(deriv_idxs)))
            for i,idx in enumerate(deriv_idxs):
                deriv_data_i = sfd._differentiate(data[:,idx:idx+1], t)
                data_deriv[:,i:i+1] = deriv_data_i

            # Create FuncMatrix
            FuncMatrix = pd.DataFrame({var: data[:, i] for i, var in enumerate(vars)})
            
            # Add squares and cubes
            for var in vars:
                FuncMatrix[f'{var}^2'] = FuncMatrix[var]**2
                FuncMatrix[f'{var}^3'] = FuncMatrix[var]**3
            for var in vars:
                FuncMatrix[f'sin({var})'] = np.sin(FuncMatrix[var])
                FuncMatrix[f'cos({var})'] = np.cos(FuncMatrix[var])
                FuncMatrix[f'exp({var})'] = np.exp(FuncMatrix[var])
                FuncMatrix[f'cosh({var})'] = np.cosh(FuncMatrix[var])
                FuncMatrix[f'tanh({var})'] = np.tanh(FuncMatrix[var])
                FuncMatrix[f'abs({var})'] = np.abs(FuncMatrix[var])
                FuncMatrix[f'sign({var})'] = np.sign(FuncMatrix[var])

            # Add cross products (up to 3rd order)
            for i, var1 in enumerate(vars):
                for j, var2 in enumerate(vars[i:]):
                    if var1 != var2:
                        FuncMatrix[f'{var1}*{var2}'] = FuncMatrix[var1] * FuncMatrix[var2]
                    for k, var3 in enumerate(vars[j:]):
                        FuncMatrix[f'{var1}*{var2}*{var3}'] = FuncMatrix[var1] * FuncMatrix[var2] * FuncMatrix[var3]

            # Create NumDiv
            NumDiv = pd.DataFrame({f'd{var}/dt': data_deriv[:, i] for i, var in enumerate(vars)})

            # Create Lambda
            Lambda = pd.DataFrame(np.array([[0.1, 0.1, 0.1] for _ in range(dim)]))

            for dim, vardotname in enumerate(dotsvarnames):
                
                p = f'./log/chaotic_wAIC2/{class_name}/{vardotname}/'
                
                if os.path.exists(f'{p}pf_{seed}.csv'):
                    print(f'exist {p}pf_{seed}.csv, skip.')
                    continue
                
                start_time = time.time()

                FuncMatrix = FuncMatrix.replace([np.nan, np.inf, -np.inf], 0)
                NumDiv = NumDiv.replace([np.nan, np.inf, -np.inf], 0)

                result = TwoPhaseInference(FuncMatrix, NumDiv, Nnodes=10, dim=dim, Dim=dim, 
                                            keep=8, SampleTime=30, batchsize=3, 
                                            Lambda=Lambda, plotstart=0.1, plotend=0.9)
                
                end_time = time.time()
                time_cost = end_time - start_time
                print('time_cost', time_cost)
                
                InferredResults, PhaseOne_series, aic_final, intercept = result
                create_dir_if_not_exist(p)
                with open(f'{p}time.txt','a') as f:
                    f.write(f'{time_cost}\n')
                
                equation = parse_inferred_results(InferredResults)
                
                # Save as CSV
                df = pd.DataFrame({'sympy_format': [equation]})
                df.to_csv(f'{p}pf_{seed}.csv', index=True)
                
                print(f"\nDimension {dim} results:")
                print(f"Inferred equation: {equation}")
                print(f"Includes constant term: {intercept}")
                print(f"Final AIC value: {aic_final[-1]}")
                
if __name__ == '__main__':
    main()