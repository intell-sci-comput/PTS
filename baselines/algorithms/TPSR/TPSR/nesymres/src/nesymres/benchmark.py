from .dclasses import Equation
#from .utils import load_dataset
import numpy as np
from sympy import sympify
import pandas as pd
import sympy
from sympy import lambdify


def evaluate_model(model_predict,
                   benchmark_path,
                   equation_idx,
                   num_test_points,
                   pointwise_acc_rtol,
                   pointwise_acc_atol):
    """
    model_predict is a callable that takes an X of shape
    (n_datapoints, n_variables) and returns scalar predictions
    """
    #dataset = load_dataset(path)
    gt_equation, num_variables, supp = get_robust_data(eq,
                                                     cfg)
    print(f'gt_equation: {gt_equation}')

    metrics = {'gt_equation': gt_equation,
               'num_variables': num_variables,}
    for iid_ood_mode in ['iid', 'ood']:
        X, y = get_data(gt_equation, num_variables, supp, num_test_points,
                        iid_ood_mode=iid_ood_mode)

        y_pred = model_predict(X)
        assert y_pred.shape == (X.shape[0],)

        if np.iscomplex(y_pred).any():
            warnings.warn('Complex values found!')
            y_pred = np.real(y_pred)

        pointwise_acc = get_pointwise_acc(y, y_pred,
                                          rtol=pointwise_acc_rtol,
                                          atol=pointwise_acc_atol)
        acc_key = _get_acc_key(iid_ood_mode)
        metrics[acc_key] = pointwise_acc

        # Drop all indices where the ground truth is NaN or +-inf
        assert y.ndim == 1
        valid_idxs = ~np.isnan(y) & ~np.isinf(y)
        metrics[f'frac_valid_{iid_ood_mode}'] = valid_idxs.sum() / num_test_points
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

    return metrics

def get_data_reject_nan(eq: Equation, cfg):
    n_attempts_max = 100
    X, y = get_data(gt_equation, num_variables, supp, num_eval_points, iid_ood_mode)
    for _ in range(n_attempts_max):
        nans = np.isnan(y)
        if not nans.any():
            break

        n_nans = nans.sum()
        X[nans], y[nans] = get_data(gt_equation, num_variables, supp, n_nans,
                                    iid_ood_mode)
    if nans.any():
        raise ValueError('Could not sample valid points for equation '
                         f'{gt_equation} supp={supp}')
    return X, y

def get_data(eq: Equation,number_of_points,  mode):
    """
    iid_ood_mode: if set to "iid", sample uniformly from the support as given
                  by supp; if set to "ood", sample from a larger support

    """
    sym = []
    vars_list = []
    for i, var in enumerate(eq.variables):
        # U means random uniform sampling.
        # Currently this is the only mode we support.
        # Decide what to do about the linspace mode.
        # assert 'U' in eq.support[var], (f'Support modes {eq.support[var].keys()} not '
        #                             f'implemented! Decide what to do about '
        #                             f'those.')
        l, h = eq.support[var]["min"], eq.support[var]["max"]
        if mode == 'iid':
            x = np.random.uniform(l, h,number_of_points)
        elif mode == 'ood':
            support_length = h - l
            assert support_length > 0
            x = np.random.uniform(l-support_length, h+support_length,
                                    number_of_points)
        else:
            raise ValueError(f'Invalid iid_ood_mode: {mode}')
        sym.append(x)
        vars_list.append(vars_list)

    X = np.column_stack(sym)
    assert X.ndim == 2
    assert X.shape[0] == number_of_points
    var = return_order_variables(eq.variables)
    y = evaluate_func(eq.expr, var, X)
    #y = lambdify(var,eq.expr)(*X.T)[:,None]
    #y = evaluate_func(gt_equation, vars_list, X)
    return X, y



def get_robust_data(eq: Equation,mode, cfg):
    n_attempts_max = 100
    X, y = get_data(eq,  eq.number_of_points, mode)
    for _ in range(n_attempts_max):
        to_replace = np.isnan(y).squeeze() | np.iscomplex(y).squeeze()
        if not to_replace.any():
            break

        n_to_replace = to_replace.sum()
        X[to_replace,:], y[to_replace] = get_data(eq,n_to_replace,mode)
    if to_replace.any():
        #get_data(eq,  eq.number_of_points, mode)
        raise ValueError('Could not sample valid points for equation '
                         f'{eq.expr} supp={eq.support}')
        
        
    return X, y

    
def load_equation(benchmark_path, equation_idx):
    df = load_data(benchmark_path)
    benchmark_row = df.loc[equation_idx]
    gt_equation = benchmark_row['eq']
    supp = eval(benchmark_row['support'])
    variables = set(supp.keys())
    eq = Equation(code=None, 
                    expr=gt_equation, 
                    coeff_dict= None, 
                    variables=variables, 
                    support=supp, 
                    valid = True,
                    number_of_points= benchmark_row['num_points'] )
    return eq

def load_data(benchmark_name):
    df = pd.read_csv(benchmark_name)
    if not all(x in df.columns for x in ["eq","support","num_points"]):
        raise ValueError("dataframe not compliant with the format. Ensure that it has eq, support and num_points as column name")
    df = df[["eq","support","num_points"]]
    return df    


def get_variables(equation):
    """ Parse all free variables in equations and return them in
    lexicographic order"""
    expr = sympy.parse_expr(equation)
    variables = expr.free_symbols
    variables = {str(v) for v in variables}
    # # Tighter sanity check: we only accept variables in ascending order
    # # to avoid silent errors with lambdify later.
    # if (variables not in [{'x'}, {'x', 'y'}, {'x', 'y', 'z'}]
    #         and variables not in [{'x1'}, {'x1', 'x2'}, {'x1', 'x2', 'x3'}]):
    #     raise ValueError(f'Unexpected set of variables: {variables}. '
    #                      f'If you want to allow this, make sure that the '
    #                      f'order of variables of the lambdify output will be '
    #                      f'correct.')

    # Make a sorted list out of variables
    # Assumption: the correct order is lexicographic (x, y, z)
    variables = sorted(variables)
    return variables

def return_order_variables(var:set):
    return sorted(list(var), key= lambda x: int(x[2:]))



def evaluate_func(func_str, vars_list, X):
    assert X.ndim == 2
    assert len(set(vars_list)) == len(vars_list), 'Duplicates in vars_list!'

    order_list = vars_list
    indeces = [int(x[2:])-1 for x in order_list]

    if not order_list:
        # Empty order list. Constant function predicted
        f = lambdify([], func_str)
        return f() * np.ones(X.shape[0])

    # Pad X with zero-columns, allowing for variables to appear in the equation
    # that are not in the ground-truth equation
    X_padded = np.zeros((X.shape[0], len(vars_list)))

    
    X_padded[:, :X.shape[1]] = X[:,:X_padded.shape[1]]
    # Subselect columns of X that corrspond to provided variables
    X_subsel = X_padded[:, indeces]

    # The positional arguments of the resulting function will correspond to
    # the order of variables in "vars_list"
    f = lambdify(vars_list, func_str)
    return f(*X_subsel.T)
