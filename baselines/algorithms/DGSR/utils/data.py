import pandas as pd
import numpy as np
from numpy import sin, cos, tan, exp, log, sinh, cosh, sqrt

def add_noise(Y, ratio, seed):
    '''add noise to data Y,
    Y shape (n,1)'''
    np.random.seed(seed)
    f_n = np.random.normal(0, 1, Y.shape[0]).reshape(-1,1)
    f_n = f_n / np.std(f_n)
    Y_noise = Y + ratio * np.sqrt(np.mean(Y**2)) * f_n
    return Y_noise

def get_benchmark_data(benchmark_name, down_sample=1000):
    df = pd.read_csv('./benchmark/benchmark.csv')
    name, dimension, use_constant, distrib, range_ls, expression = df[
        df['name'] == benchmark_name].iloc[0]

    range_ls = eval(range_ls)

    assert len(range_ls) == 1 or len(range_ls) == dimension
    if len(range_ls) == 1 and dimension > 1:
        range_ls = range_ls * dimension
    if use_constant == 0:
        use_constant = False
    elif use_constant == 1:
        use_constant = True
    else:
        raise ValueError('use_constant should be 0 or 1')

    variables_name = []
    for i in range(dimension):
        variables_name.append('x{}'.format(i+1))

    X = generate_X(range_ls, down_sample, distrib)
    for i in range(X.shape[1]):
        globals()['x{}'.format(i+1)] = X[:, i]
    Y = eval(expression).reshape(-1, 1)
    return X, Y, use_constant, expression, variables_name


def generate_X(ranges, down_sample, distrib='U'):

    num_dims = len(ranges)
    dims = [n_points for _, _, n_points in ranges]
    num_points = 1
    for dim in dims:
        num_points *= dim
    n = min(num_points, down_sample)
    points = np.empty((n, num_dims))

    for i in range(n):
        if distrib == 'U':
            steps = [np.sort(np.random.uniform(start, stop, size=n_points))
                     for start, stop, n_points in ranges]
        elif distrib == 'E':
            steps = [np.linspace(start, stop, num=n_points)
                     for start, stop, n_points in ranges]
        else:
            raise ValueError('distrib should be U or E')
        for j in range(num_dims):
            step = steps[j]
            val = np.random.choice(step)
            points[i, j] = val

    return points


def get_dynamic_data(dataset_name, file_name):
    '''
    return dataset df, variables name and target name

    Example
    =======

    >>> df, variables_name, target_name = get_dynamic_data('ball','Baseball_train')
    >>> variables_name
    >>> ['t']
    >>> target_name
    >>> 'h'
    '''
    df = pd.read_csv('./data/'+dataset_name+'/'+file_name+'.csv', header=None)
    # NOTE: If use your own dataset, the column name cannot be `C` or `B`,
    # because it's used as constant symbol in regressor
    if dataset_name == 'ball':
        names = ['t', 'h']
        target_name = 'h'
    elif dataset_name == 'dp':
        names = ['x1', 'x2', 'w1', 'w2', 'wdot', 'f']
        target_name = 'f'
    elif dataset_name == 'lorenz':
        names = ['x', 'y', 'z', 'f']
        target_name = 'f'
    elif dataset_name == 'emps':
        names = ['q', 'qdot', 'qddot', 'tau']
        target_name = 'qddot'
    elif dataset_name == 'silverbox':
        names = ['u', 'y', 'ydot', 'yddot']
        target_name = 'yddot'
    else:
        raise ValueError(
            'dataset_name should be `ball`, `dp`, `lorenz`, `silverbox`, or `emps`')

    df.columns = names
    variables_name = names.copy()
    variables_name.remove(target_name)

    return df, variables_name, target_name

def expr_to_Y_pred(expr_sympy, X, variables):
    functions = {
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        'arcsin': np.arcsin,
        'arccos': np.arccos,
        'arctan': np.arctan,
        'sign': np.sign,
        'e': np.exp(1),
        'pi': np.pi,
    }
    expr_str = str(expr_sympy)
    values = {variables[j]: X[:, j] for j in range(X.shape[1])}
    pred = eval(expr_str.lower(), functions, values)
    return pred