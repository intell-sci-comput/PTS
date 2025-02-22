import argparse
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import hydra
from csem_exptrack import process, utils
import json
from nesymres import benchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dirs', type=str, required=True, nargs='+')

    args = parser.parse_args()
    return args


def rename_benchmark(benchmark_name):
    """We updated our benchmark name from "ours" to "ours-nc"."""
    if benchmark_name == 'ours':
        return 'ours-nc'
    return benchmark_name


def standardize_equation(equation):
    """
    Replace x1 -> x, x2 -> y, x3 -> z in order to work with equivalent formats.
    Also, add 0 times all variables in order to make all variables appear always
    """
    if equation is None or (not isinstance(equation, str) and np.isnan(equation)):
        return None
    equation = equation.replace('x1', 'x')
    equation = equation.replace('x2', 'y')
    equation = equation.replace('x3', 'z')
    return equation


def patch_benchmark_name(args_df):
    """For experiments before 2021-02-03, there was no key 'benchmark_name'.
    They were all using the AI Feynman dataset
    """
    output_dirs = args_df.output_dir
    output_dir = output_dirs[0]
    date = Path(output_dir).parent.name.split('_')[0]
    print(f'date: {date}')
    if datetime.strptime(date, '%Y-%m-%d') < datetime(2021, 2, 3):
        print('Patching benchmark_name!')
        args_df['benchmark_name'] = 'ai_feymann'

@hydra.main(config_name="collect_results")
def main(cfg):
    return collect_results(cfg)


def collect_results(cfg):
    loader = process.file_loader.FileLoader("results.json")
    df = loader.load_folder(hydra.utils.to_absolute_path("fin_res_v2")).T    
    #f = raw_df.loc[[0,("other","test_path"),("other","name"),("other", "eq"),("other", "benchmark_name") ]]
     
    eqs = list(df.loc[:,[("other", "equation_idx")]].values.reshape(-1))
    
    res = []
    for i in range(len(df)):
        with open(df['path'].iloc[i].values[0]) as json_file:
            json_data = json.load(json_file)
            res.append(json_data)
    best_eq = [x["equation"][0] if x["equation"] else None for x in res]
    duration = [x["duration"] for x in res]
    df.loc[:,"pred_eq"] = best_eq
    df.loc[:,"duration"] = duration
    df.index = eqs

    eval_rows = []

    for idx, df_row in tqdm(df.iterrows(), desc='Evaluate equations...'):
        if df_row.pred_eq[0]:
            assert getattr(df_row, 'model_path', None) is None
            metrics = evaluate_equation(
                pred_equation=df_row.pred_eq[0],
                benchmark_name=hydra.utils.to_absolute_path(df_row.loc[[("other", "benchmark_path")]][0]),
                equation_idx=df_row.loc[[("other", "equation_idx")]][0],
                cfg=cfg)
        else:
            metrics = {}
        # else:
        #     model_path = reroute_path(df_row.model_path, df_row.output_dir,
        #         root_dirs)
        #     metrics = evaluate_sklearn(
        #         model_path=model_path,
        #         benchmark_name=df_row.benchmark_name,
        #         equation_idx=df_row.equation_idx,
        #         num_test_points=cfg.NUM_TEST_POINTS,
        #         pointwise_acc_rtol=cfg.POINTWISE_ACC_RTOL,
        #         pointwise_acc_atol=cfg.POINTWISE_ACC_ATOL
        #     )
        eval_row = df_row.to_dict()
        eval_row.update(metrics)
        eval_rows.append(eval_row)

    
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv('eval_df_v2.csv')

    return eval_df #, root_dirs


def get_pointwise_acc(y_true, y_pred, rtol, atol):
    # r = roughly_equal(y_true, y_pred, rel_threshold)
    r = np.isclose(y_true, y_pred, rtol=rtol, atol=atol, equal_nan=True)
    return r.mean()


def evaluate_equation(pred_equation, benchmark_name, equation_idx, cfg):
    """Evaluate equation `pred_equation` given as a string."""

    def model_predict(X):
        pred_variables = benchmark.get_variables(pred_equation)
        # assert len(pred_variables) <= X.shape[1]
        #y = lambdify(list(eq.variables),eq.expr)(*X.T)[:,None]
        return benchmark.evaluate_func(pred_equation, pred_variables, X)

    return evaluate_model(model_predict, benchmark_name, equation_idx,
                          cfg)


def evaluate_sklearn(model_path, benchmark_name, equation_idx,
                      num_test_points, pointwise_acc_rtol,
                      pointwise_acc_atol):
    """Evaluate sklearn model at model_path."""

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    def model_predict(X):
        return model.predict(X)

    return evaluate_model(model_predict, benchmark_name, equation_idx,
                          num_test_points,
                          pointwise_acc_rtol, pointwise_acc_atol)


def evaluate_model(model_predict,
                   benchmark_path,
                   equation_idx,
                   cfg):
    """
    model_predict is a callable that takes an X of shape
    (n_datapoints, n_variables) and returns scalar predictions
    """
    
    eq = benchmark.load_equation(benchmark_path,equation_idx)
    print(f'gt_equation: {eq.expr}')

    metrics = {'gt_equation': eq.expr,
               'num_variables': len(eq.variables),}
    for iid_ood_mode in ['iid', 'ood']:
        #get_data(eq,  eq.number_of_points, mode, cfg)
        X, y = benchmark.get_data(eq, cfg.num_test_points, mode=iid_ood_mode) #Think about robust nan
        y_pred = model_predict(X)
        assert y_pred.shape == (X.shape[0],)

        if np.iscomplex(y_pred).any():
            warnings.warn('Complex values found!')
            y_pred = np.real(y_pred)

        pointwise_acc = get_pointwise_acc(y, y_pred,rtol=cfg.pointwise_acc_rtol,atol=cfg.pointwise_acc_atol)
        acc_key = _get_acc_key(iid_ood_mode,cfg)
        metrics[acc_key] = pointwise_acc
        # Drop all indices where the ground truth is NaN or +-inf
        #y = y.squeeze() #Points, Y_dim
        assert y.ndim == 1 #Points 
        #y.squeeze()
        valid_idxs = ~np.isnan(y) & ~np.isinf(y)
        metrics[f'frac_valid_{iid_ood_mode}'] = valid_idxs.sum() / cfg.num_test_points
        
        y = y[valid_idxs]
        y_pred = y_pred[valid_idxs]
        assert y.shape[0] == valid_idxs.sum()

        # Replace NaN or infinity prediction
        replace_prediction_idxs = np.isnan(y_pred) | np.isinf(y_pred)
        metrics[f'frac_replace_{iid_ood_mode}'] = replace_prediction_idxs.sum() / y.shape[0]
        y_pred[replace_prediction_idxs] = 0.0

        # Add default metrics
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        metrics[f'r2_{iid_ood_mode}'] = r2
        metrics[f'mse_{iid_ood_mode}'] = mse

    # if equation_idx == 30:
    #     breakpoint()
    return metrics


def _get_acc_key(iid_ood_mode,cfg):
    return (f'pointwise_acc_r{cfg.pointwise_acc_rtol:.2}_'
            f'a{cfg.pointwise_acc_atol:.2}_{iid_ood_mode}')


def plot_durations(combined_df):
    ax = sns.stripplot('nesymres_beam_size', 'duration', data=combined_df, size=3)
    ax.set_yscale('log')
    ax.set_ylabel('duration (seconds)')
    sns.despine(ax=ax)
    ax.grid(axis='y')
    ax.grid(axis='y', which='minor', alpha=0.2)
    plt.show()


if __name__ == '__main__':
    eval_df = main()