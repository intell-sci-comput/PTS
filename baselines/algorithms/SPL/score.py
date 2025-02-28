import numpy as np
from numpy import *
from sympy import simplify, expand
from scipy.optimize import minimize
from contextlib import contextmanager
import threading
import _thread
import time


class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def simplify_eq(eq):
    return str(expand(simplify(eq)))


def prune_poly_c(eq):
    '''
    if polynomial of C appear in eq, reduce to C for computational efficiency. 
    '''
    eq = simplify_eq(eq)
    if 'C**' in eq:
        c_poly = ['C**'+str(i) for i in range(10)]
        for c in c_poly:
            if c in eq: eq = eq.replace(c, 'C')
    return simplify_eq(eq)


def score_with_est(eq, tree_size, data, t_limit = 1.0, eta=0.999):
    """
    Calculate reward score for a complete parse tree 
    If placeholder C is in the equation, also excute estimation for C
    Reward = 1 / (1 + MSE) * Penalty ** num_term

    Parameters
    ----------
    eq : Str object.
        the discovered equation (with placeholders for coefficients). 
    tree_size : Int object.
        number of production rules in the complete parse tree. 
    data : 2-d numpy array.
        measurement data, including independent and dependent variables (last row). 
    t_limit : Float object.
        time limit (seconds) for ssingle evaluation, default 1 second. 
        
    Returns
    -------
    score: Float
        discovered equations. 
    eq: Str
        discovered equations with estimated numerical values. 
    """

    ## define independent variables and dependent variable
    num_var = data.shape[0] - 1
    if num_var <= 3: ## most cases ([x], [x,y], or [x,y,z])
        current_var = 'x'
        for i in range(num_var):
            globals()[current_var] = data[i, :]
            current_var = chr(ord(current_var) + 1)
        globals()['f_true'] = data[-1, :]
    else:            ## currently only double pendulum case has more than 3 independent variables
        globals()['x1'] = data[0, :]
        globals()['x2'] = data[1, :]
        globals()['w1'] = data[2, :]
        globals()['w2'] = data[3, :]
        globals()['wdot'] = data[4, :]
        globals()['f_true'] = data[5, :]


    ## count number of numerical values in eq
    c_count = eq.count('C')
    # start_time = time.time()
    with time_limit(t_limit, 'sleep'):
        try: 
            if c_count == 0:       ## no numerical values
                f_pred = eval(eq)
            elif c_count >= 10:    ## discourage over complicated numerical estimations
                return 0, eq
            else:                  ## with numerical values: coefficient estimationwith Powell method

                # eq = prune_poly_c(eq)
                c_lst = ['c'+str(i) for i in range(c_count)]
                for c in c_lst: 
                    eq = eq.replace('C', c, 1)

                def eq_test(c):
                    for i in range(len(c)): globals()['c'+str(i)] = c[i]
                    return np.linalg.norm(eval(eq) - f_true, 2)

                x0 = [1.0] * len(c_lst)
                c_lst = minimize(eq_test, x0, method='Powell', tol=1e-6).x.tolist() 
                c_lst = [np.round(x, 4) if abs(x) > 1e-2 else 0 for x in c_lst]
                eq_est = eq
                for i in range(len(c_lst)):
                    eq_est = eq_est.replace('c'+str(i), str(c_lst[i]), 1)
                eq = eq_est.replace('+-', '-')
                f_pred = eval(eq)
        except: 
            return 0, eq

    r = float(eta ** tree_size / (1.0 + np.linalg.norm(f_pred - f_true, 2) ** 2 / f_true.shape[0]))

    # run_time = np.round(time.time() - start_time, 3)
    # print('runtime :', run_time,  eq,  np.round(r, 3))
    
    return r, eq

