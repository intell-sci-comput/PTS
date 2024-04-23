import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sympy as sp
import os
from contextlib import contextmanager
import threading
import _thread
import time

####### NOTE: TimeoutException can not be used on Windows platform #######

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
    
def insert_B_on_Add(expr_sympy):

    cnt_B = 0

    def do(x):
        B = sp.Symbol('B')
        nonlocal cnt_B
        cnt_B += 1
        return x.func(*(x.args + (B,)))
    expr_sympy = expr_sympy.replace(lambda x: x.is_Add, lambda x: do(x))
    return expr_sympy

def condense(eq, *x):
    """collapse additive/multiplicative constants into single
    variables, returning condensed expression and replacement
    values.
    
    https://stackoverflow.com/questions/71315789/
    """
    reps = {}
    con = sp.numbered_symbols('c')
    free = eq.free_symbols

    def c():
        while True:
            rv = next(con)
            if rv not in free:
                return rv

    def do(e):
        i, d = e.as_independent(*x)
        if not i.args:
            return e
        return e.func(reps.get(i, reps.setdefault(i, c())), d)
    rv = eq.replace(lambda x: x.is_Add or x.is_Mul, lambda x: do(x))
    reps = {v: k for k, v in reps.items()}
    keep = rv.free_symbols & set(reps)
    reps = {k: reps[k].xreplace(reps) for k in keep}
    return rv, reps


def densify(expr_c, variables):
    variables_sympy = [sp.Symbol(v) for v in variables]
    expr_c_sympy = sp.sympify(expr_c)
    expr_c_dense_sympy, dense_dict = condense(expr_c_sympy, *variables_sympy)
    value_ls = []
    name_ls = []
    for key, value in zip(dense_dict.keys(), dense_dict.values()):
        for f in value.free_symbols:
            value = value.subs(f, 1.)
        value_ls.append(value)
        name_ls.append(key)
    for atom in expr_c_dense_sympy.atoms():
        if 'C' in str(atom):
            value_ls.append(1.)
            name_ls.append(atom)
    dict_final = {}
    for i in range(len(value_ls)):
        dict_final[name_ls[i]] = value_ls[i]
    return expr_c_dense_sympy, dict_final

def remove_bias(expr):
    constants = [term for term in expr.as_ordered_terms() if term.is_constant()]
    bias = sum(constants)
    expr_without_bias = expr - bias
    return expr_without_bias, bias

def finallize_const_name(expr_dense_sympy, dict_final, add_bias=True):
    cnt_c = 0
    dict_final_final = {}
    for atom in expr_dense_sympy.atoms():
        if 'C' in str(atom) or 'c' in str(atom) or 'B' in str(atom):
            new_atom = sp.Symbol('a{}'.format(cnt_c))
            expr_dense_sympy = expr_dense_sympy.subs(atom, new_atom)
            cnt_c += 1
            dict_final_final[new_atom] = dict_final[atom]
    if add_bias:
        if expr_dense_sympy.func is not sp.core.add.Add:
            new_atom = sp.Symbol('a{}'.format(cnt_c))
            expr_dense_sympy += sp.sympify(new_atom)
            dict_final_final[new_atom] = 0.0
            cnt_c += 1
    return expr_dense_sympy, dict_final_final


def replace_c_with_a(expr_dense_sympy):
    for atom in expr_dense_sympy.atoms():
        str_atom = str(atom)
        if 'a' == str_atom[0] and len(str_atom) >= 2 and str_atom[1:].isdigit():
            new_atom = sp.Symbol('C{}'.format(str_atom[1:]))
            expr_dense_sympy = expr_dense_sympy.subs(atom, new_atom)
    return expr_dense_sympy


def is_const(expr_sympy):
    val = expr_sympy.n(1)
    if isinstance(val, sp.core.numbers.Float) and (not expr_sympy.is_Number):
        return True
    else:
        return False


def replace_evaluatable(expr):
    replace_map = {}
    for subexpr in expr.find(is_const):
        val = subexpr.evalf()
        replace_map[subexpr] = val
    return expr.subs(replace_map, simultaneous=True)


def to_C_expr(expr, variables):
    expr_num = replace_evaluatable(expr)
    expr_num = str(expr_num)
    ops = ['sin', 'cos', 'tan',
           'log', 'asin', 'acos', 'atan', 'sign']
    for op in ops:
        expr_num = expr_num.replace(op, 'C*{}'.format(op))
    for variable in variables:
        expr_num = re.sub(
            r'(?<![a-zA-Z]){}(?![a-zA-Z])'.format(variable), r'(C*{})'.format(variable), expr_num)

    cnt_C = 0
    def replace_C(matched):
        nonlocal cnt_C
        cnt_C += 1
        return 'C{}'.format(cnt_C-1)
    expr_num = re.sub(r'C', replace_C, expr_num)
    return expr_num


def replace_B(expr_c_sympy):
    cnt_B = 0

    def replace_C(matched):
        nonlocal cnt_B
        cnt_B += 1
        return 'B{}'.format(cnt_B-1)
    expr_c_sympy = re.sub(r'B', replace_C, expr_c_sympy)
    return expr_c_sympy, cnt_B


def get_expr_C_and_C0(expr, variables, add_bias=True):
    expr_sympy = sp.sympify(expr)
    expr_c = to_C_expr(expr_sympy, variables)
    expr_c_sympy = sp.sympify(expr_c)
    expr_c_sympy, dict_c = densify(expr_c_sympy, variables) 
    expr_c_sympy, bias = remove_bias(expr_c_sympy)
    if add_bias:
        expr_c_sympy = insert_B_on_Add(expr_c_sympy)
    expr_c_sympy_str = str(expr_c_sympy)
    expr_c_sympy_str, cnt_B = replace_B(expr_c_sympy_str)
    for i in range(cnt_B):
        dict_c[sp.Symbol('B{}'.format(i))] = bias
    expr_c_sympy = sp.sympify(expr_c_sympy_str)
    expr_dense_sympy, dict_final = finallize_const_name(
        expr_c_sympy, dict_c, add_bias=True)
    expr_final_sympy = replace_c_with_a(expr_dense_sympy)
    C0 = np.array(list(dict_final.values()))

    return str(expr_final_sympy), C0

def my_equals_struct(expr1, expr2, is_positive, variables):
    
    expr1, C0_1 = get_expr_C_and_C0(set_real(sp.sympify(str(expr1)), is_positive).expand(), variables)
    expr2, C0_2 = get_expr_C_and_C0(set_real(sp.sympify(str(expr2)), is_positive).expand(), variables)
    expr1 = sp.sympify(expr1)
    expr2 = sp.sympify(expr2)
    
    for i in range(len(C0_1)):
        expr1 = expr1.subs(sp.Symbol('C{}'.format(i)), sp.Symbol('C'))
    for i in range(len(C0_2)):
        expr2 = expr2.subs(sp.Symbol('C{}'.format(i)), sp.Symbol('C'))
    
    print('\t => ',expr1, expr2)
    if expr1 == expr2:
        return True
    else:
        return False

def set_real(expr_c_sympy, is_positive):
    for var in expr_c_sympy.free_symbols:
        expr_c_sympy = expr_c_sympy.subs(
            var, sp.Symbol(str(var), real=True, positive=is_positive))
    return expr_c_sympy

def prun_constant(expr_num_sympy, n_digits=6):
    epsilon = 10.0**(-n_digits)
    for atom in expr_num_sympy.atoms():
        if isinstance(atom, sp.core.numbers.Float):
            if abs(atom) < epsilon:
                expr_num_sympy = sp.sympify(
                    expr_num_sympy.subs(atom, sp.sympify('0')))
            else:
                expr_num_sympy = expr_num_sympy.subs(
                    atom, round(atom, n_digits))

    return expr_num_sympy

def symgp_variables_shift(expr_str_symgp):
    for i in range(10,-1,-1):
        expr_str_symgp = expr_str_symgp.replace('x{}'.format(i),'x{}'.format(i+1))
    return expr_str_symgp

def is_symbolic_same(expr_str1, expr_str2, is_positive, variables):
    
    expr_sympy1 = sp.sympify(expr_str1)
    expr_sympy2 = sp.sympify(expr_str2)
    
    print('comparing',expr_sympy1,expr_sympy2)

    expr_sympy1 = prun_constant(expr_sympy1, n_digits=2)
    expr_sympy2 = prun_constant(expr_sympy2, n_digits=2)

    l = 30
    if len(expr_sympy1.free_symbols) != len(expr_sympy2.free_symbols) or\
        expr_sympy1.free_symbols != expr_sympy2.free_symbols:
        return False
    
    if my_equals_struct(expr_sympy1, expr_sympy2, is_positive, variables):
        print(str(expr_sympy1).rjust(l),' =sym= ', str(expr_sympy2).ljust(l))
        logging(str(expr_sympy1).rjust(l)+' =sym= '+str(expr_sympy2).ljust(l))
        return True
    else:
        print(str(expr_sympy1).rjust(l),'       ', str(expr_sympy2).ljust(l))
        logging(str(expr_sympy1).rjust(l)+'       '+str(expr_sympy2).ljust(l))
        return False
    
def is_symbolic_success(expr_str, expr_str_gt, is_positive, variables):
    t_limit = 3.0
    try:
        with time_limit(t_limit, 'sleep'):
            if is_symbolic_same(expr_str, expr_str_gt, is_positive, variables):
                return True
            else:
                return False
    except TimeoutException:
        return False
    except Exception as e:
        print(e)
        return False
        
def is_symbolic_success_se(se, benchmark_name, xyzformat=False):
    p = './benchmark/dysts.csv'
    df_benchmark = pd.read_csv(p)
    df_benchmark = df_benchmark[df_benchmark['name']==benchmark_name]
    try:
        expr_str_gt = df_benchmark['expression'].iloc[0]
        params_dict = eval(df_benchmark['params'].iloc[0])
        n_variables = df_benchmark['dimension'].iloc[0]
        try:
            expr_str_gt = sp.sympify(expr_str_gt).subs(params_dict)
        except:
            print(expr_str_gt)
            print(params_dict)
            raise ValueError
        
    except IndexError:
        print(df_benchmark['expression'])
        print('benchmark_name',benchmark_name)
        exit()
    
    if n_variables == 3:
        variables = ['x0','x1','x2']
    else:
        variables = ['x0','x1','x2','x3']
    

    is_positive = False
    for expr_str in se:
        # expr_str = expr_str[1:-1]
        if type(expr_str) == float or type(expr_str) == int:
            pass
        elif expr_str[0] == '[' and expr_str[-1] == ']':
            expr_str = expr_str[1:-1]
            
        if xyzformat:
            if xyzformat == 'psrn':
                expr_str = sp.sympify(expr_str).subs({'x':'x0','y':'x1','z':'x2','w':'x3'})
            
        if is_symbolic_success(expr_str, expr_str_gt, is_positive, variables):
            return True
    return False

def logging(str):
    path = './log_symbolic_compare/'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+'log.txt', 'a') as f:
        f.write(str+'\n')


p = './log/chaotic_symbolic_recovery/'

if not os.path.exists(p):
    os.mkdir(p)
    

mode_ls = ['psrn']

for mode in mode_ls:
    
    colname = 'expr_str'
    is_other_log = False
        

    log_dir = './log/chaotic/'
    file_name_filter = None
    
    l = os.listdir(log_dir)
    
    if file_name_filter is not None:
        l = [_ for _ in l if file_name_filter in _]
    
    print(l)

    df = pd.DataFrame(None,columns=['benchmark_name','vardot','seed','success','timecost'])

    n = len(l)

    idx = 0
    for dir_name in l:
        if not is_other_log:
            benchmark_name = dir_name
        else:
            lll = dir_name.replace(file_name_filter,'').split('_')
            benchmark_name = lll[0]
            vardot_name = lll[1]
            seed = int(lll[2])
        
        idx += 1
        print(idx / n, idx, n ,'<'*10)

        dir = log_dir + benchmark_name
        
        ll = os.listdir(dir)
        
        print(ll)
        
        for vardot in ll:
            print('dir',dir)
            dirvar = dir + '/' + vardot
            print('dirvar',dirvar)
            time_file_name = 'time.txt'
            time_ls = []
            with open(dirvar + '/' + time_file_name,'r') as f:
                for line in f.readlines():
                    time_ls.append(float(line))
            print('time_ls',time_ls)
            
            seed = 0
            for i, file_name in enumerate(os.listdir(dirvar)):
                if file_name[-3:] == 'csv':
                    try:
                        print('seed',seed)
                        print('file_name',file_name)
                        
                        df_log = pd.read_csv(dirvar + '/' + file_name)
                        success = is_symbolic_success_se(df_log[colname], benchmark_name + '_' + vardot, xyzformat='psrn')
                        df = df.append({'benchmark_name':benchmark_name,'vardot':vardot,'seed':seed,'success':success,'timecost':time_ls[seed]},ignore_index=True)
                        seed += 1
                    except FileNotFoundError:
                        print('not found',dirvar + '/' + file_name)
                        pass

        print('df',df) 
        df_grouped = df.groupby(['benchmark_name','vardot']).agg({'success':['mean','count'],'timecost':['mean','std']})
        
        print('df_grouped',df_grouped)
        df_grouped.columns = df_grouped.columns.to_flat_index() # flatten
        df_all = df_grouped.iloc[:,[0,1,2,3]]

        df_all.columns = ['success_symbolic','count','timecost_mean','timecost_std']
        # df_all.to_csv(p+'{}.csv'.format(mode))
        df_all.to_csv(p+'{}_stats.csv'.format(mode))
