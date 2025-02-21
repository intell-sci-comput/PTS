import pandas as pd
import numpy as np
from numpy import sin, cos, tan, exp, log, sinh, cosh, sqrt

def add_noise(Y, ratio, seed):
    '''add noise to data Y,
    Y shape (n,1)'''
    # np.random.seed(seed)
    # f_n = np.random.normal(0, 1, Y.shape[0]).reshape(-1,1)
    # f_n = f_n / np.std(f_n)
    # Y_noise = Y + ratio * np.sqrt(np.mean(Y**2)) * f_n
    # return Y_noise
    
    np.random.seed(seed)
    
    Y_std = Y.std(axis=0)
    noise = np.random.normal(0, Y_std, Y.shape)  
    noise = noise * ratio
    
    noisy_Y = Y + noise
    return noisy_Y


def get_benchmark_data(benchmark_file, benchmark_name, down_sample=1000):
    df = pd.read_csv('./benchmark/' + benchmark_file)
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

    if distrib == 'U':
        for i in range(n):
            steps = [np.sort(np.random.uniform(start, stop, size=n_points))
                        for start, stop, n_points in ranges]
            for j in range(num_dims):
                step = steps[j]
                val = np.random.choice(step)
                points[i, j] = val
    
    elif distrib == 'E':
        if down_sample < n * num_dims:
            raise ValueError('E distrib not support down_sample < n * num_dims')
        steps = [np.linspace(start, stop, num=n_points)
                 for start, stop, n_points in ranges]
        points = np.array(np.meshgrid(*steps)).T.reshape(-1, num_dims)
        
    else:
        raise ValueError('distrib should be U or E')
    
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
    elif dataset_name == 'roughpipe':
        names = ['l','y','k']
        target_name = 'y'
    elif dataset_name == 'cells':
        names = ['dh1','dh2','dh3','alpha','eta','p120','zo1','sxx']
        target_name = 'sxx'
    elif dataset_name == 'funding':
        names = ['att','sti','gdp','rd','ppp','rate']
        target_name = 'rate'
    elif dataset_name == 'gwave':
        names = ['t','y']
        target_name = 'y'
    elif dataset_name == 'pulse':
        names = ['t','y']
        target_name = 'y'
    elif dataset_name == 'invpend':
        names = ['v1','v2','v3','v4']
        target_name = 'v4'
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
    values = {variables[j]: X[:, j:j+1] for j in range(X.shape[1])}
    pred = eval(expr_str.lower(), functions, values) * np.ones((X.shape[0], 1))
    return pred

def select_best_expr_from_pareto_front(expr_sympy_ls, X_test, Y_test, variables):
    mse_ls = []
    for expr_sympy in expr_sympy_ls:
        Y_test_pred = expr_to_Y_pred(expr_sympy, X_test, variables)
        mse = np.mean((Y_test_pred - Y_test) ** 2)
        if np.isnan(mse):
            mse = 1e99
        mse_ls.append(mse)
    idx = np.argmin(mse_ls)
    return expr_sympy_ls[idx], mse