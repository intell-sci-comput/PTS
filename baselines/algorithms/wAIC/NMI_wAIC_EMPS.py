import os
import gc
import numpy as np
import time
import pandas as pd
from utils.data import get_dynamic_data
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
    n_seeds = n_runs


    for seed in range(n_seeds):
        df, variables_name, target_name = get_dynamic_data('emps', 'emps')

        # Select the first half of the data as train set
        df = df.iloc[:len(df) // 2, :]

        np.random.seed(seed)

        p = './log/EMPS/'
        if os.path.exists(f'{p}pf_{seed}.csv'):
            print(f'exist {p}pf_{seed}.csv, skip.')
            continue

        start_time = time.time()

        # Create FuncMatrix
        FuncMatrix = pd.DataFrame({var: df[var] for var in variables_name})

        # Add squares and cubes
        for var in variables_name:
            FuncMatrix[f'{var}^2'] = FuncMatrix[var] ** 2
            FuncMatrix[f'{var}^3'] = FuncMatrix[var] ** 3

        # Add sin, cos, exp, cosh, tanh, abs, sign
        for var in variables_name:
            FuncMatrix[f'sign({var})'] = np.sign(FuncMatrix[var])

        # Add cross products (up to 3rd order)
        for i, var1 in enumerate(variables_name):
            for j, var2 in enumerate(variables_name[i:]):
                if var1 != var2:
                    FuncMatrix[f'{var1}*{var2}'] = FuncMatrix[var1] * FuncMatrix[var2]
                for k, var3 in enumerate(variables_name[j:]):
                    FuncMatrix[f'{var1}*{var2}*{var3}'] = FuncMatrix[var1] * FuncMatrix[var2] * FuncMatrix[var3]

        # Create NumDiv
        NumDiv = pd.DataFrame({target_name: df[target_name]})
        print('target_name',target_name)
        print('variables_name',variables_name)
        print('NumDiv',NumDiv.shape)
        # Create Lambda
        Dim = len(variables_name)
        Lambda = pd.DataFrame(np.array([[0.1, 0.1, 0.1] for _ in range(Dim)]))

        FuncMatrix = FuncMatrix.replace([np.nan, np.inf, -np.inf], 0)
        NumDiv = NumDiv.replace([np.nan, np.inf, -np.inf], 0)
        
        print('FuncMatrix',FuncMatrix.shape)

        result = TwoPhaseInference(FuncMatrix, NumDiv, Nnodes=10, dim=0, Dim=Dim, 
                                    keep=8, SampleTime=60, batchsize=3, 
                                    Lambda=Lambda, plotstart=0.1, plotend=0.9)

        end_time = time.time()
        time_cost = end_time - start_time
        print('time_cost', time_cost)

        InferredResults, PhaseOne_series, aic_final, intercept = result
        create_dir_if_not_exist(p)
        with open(f'{p}time.txt', 'a') as f:
            f.write(f'{time_cost}\n')

        equation = parse_inferred_results(InferredResults)

        # Save as CSV
        df = pd.DataFrame({'sympy_format': [equation]})
        df.to_csv(f'{p}pf_{seed}.csv', index=True)

        print(f"\nResults:")
        print(f"Inferred equation: {equation}")
        print(f"Includes constant term: {intercept}")
        print(f"Final AIC value: {aic_final[-1]}")

if __name__ == '__main__':
    main()