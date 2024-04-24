import math
import itertools
import re

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sympy
from sympy import Eq

from scipy.special import comb
import scipy.optimize as opt

from model.operators import (Identity_op,
                             Sin_op,
                             Cos_op,
                             Exp_op,
                             Log_op,
                             Neg_op,
                             Inv_op,
                             Add_op,
                             Mul_op,
                             Sub_op,
                             Div_op,
                             SemiDiv_op,
                             SemiSub_op)

from model.operators import (Sign_op,
                             Pow2_op,
                             Pow3_op,
                             Pow_op,
                             Sigmoid_op,
                             Abs_op,
                             Cosh_op,
                             Tanh_op,
                             Sqrt_op)

from model.models import PSRN

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device('cuda')


def insert_B_on_Add(expr_sympy):
    """add bias terms to Add node 

    Examples
    ========

    >>> expr1 + expr2 -> expr1 + expr2 + B

    """
    cnt_B = 0

    def do(x):
        B = sympy.Symbol('B')
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
    con = sympy.numbered_symbols('c')
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
    variables_sympy = [sympy.Symbol(v) for v in variables]
    expr_c_sympy = sympy.sympify(expr_c)
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


def finallize_const_name(expr_dense_sympy, dict_final, add_bias=True):
    cnt_c = 0
    dict_final_final = {}
    for atom in expr_dense_sympy.atoms():
        if 'C' in str(atom) or 'c' in str(atom) or 'B' in str(atom):
            new_atom = sympy.Symbol('a{}'.format(cnt_c))
            expr_dense_sympy = expr_dense_sympy.subs(atom, new_atom)
            cnt_c += 1
            dict_final_final[new_atom] = dict_final[atom]
    if add_bias:
        if expr_dense_sympy.func is not sympy.core.add.Add:
            new_atom = sympy.Symbol('a{}'.format(cnt_c))
            expr_dense_sympy += sympy.sympify(new_atom)
            dict_final_final[new_atom] = 0.0
            cnt_c += 1
    return expr_dense_sympy, dict_final_final


def replace_c_with_a(expr_dense_sympy):
    # a0 + a1*qdot + a2*tau + a4*qdot*log(Abs(a3*qdot))
    '''
    replace all a constant to C constant in expressioin

    Examples
    ========
    Input:
    >>> a0 * sin(x) + a1
    Output:
    >>> C0 * sin(x) + C1

    '''
    for atom in expr_dense_sympy.atoms():
        str_atom = str(atom)
        if 'a' == str_atom[0] and len(str_atom) >= 2 and str_atom[1:].isdigit():
            new_atom = sympy.Symbol('C{}'.format(str_atom[1:]))
            expr_dense_sympy = expr_dense_sympy.subs(atom, new_atom)
    return expr_dense_sympy


def is_const(expr_sympy):
    '''judge whether sub expression is const

    Examples
    ========
    Input:
    >>> exp(2)
    Output:
    >>> True

    Input:
    >>> exp(x)
    Output:
    >>> False

    '''
    val = expr_sympy.n(1)
    if isinstance(val, sympy.core.numbers.Float) and (not expr_sympy.is_Number):
        return True
    else:
        return False


def replace_evaluatable(expr):
    '''
    Find the result of all the evaluatable parts of the expression
    '''
    replace_map = {}
    for subexpr in expr.find(is_const):
        val = subexpr.evalf()
        replace_map[subexpr] = val
    return expr.subs(replace_map, simultaneous=True)

def replace_exponent(expression):
    pattern = r'\*\*(\((.*?)\)|(\d+(\.\d+)?))'

    def replace_func(match):
        exp = match.group(2) or match.group(3)
        if exp.strip().replace(".", "").isdigit():
            return f"**(C*{exp})"
        else:
            return f"**(C*({exp}))"

    replaced_expression = re.sub(pattern, replace_func, expression)
    
    pattern = r'\*\*(\d+(\.\d+)?)(?!\d)'
    def replace_func(match):
        exp = match.group(1)
        if exp.strip().replace(".", "").isdigit():
            return f"**(C*{exp})"
        else:
            return f"**(C*({exp}))"
    
    replaced_expression = re.sub(pattern, replace_func, replaced_expression)
    
    return replaced_expression


def to_C_expr(expr, variables, use_replace_exponent=False):
    '''
    1. Add a constant C to the front of all operators except exp and Abs
    2. Replace all variables with c* variables
    '''
    expr_num = replace_evaluatable(expr)
    expr_num = str(expr_num)
    
    if use_replace_exponent:
        expr_num = replace_exponent(expr_num)
        print('replaced:',expr_num)
    
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
    '''
    Replace all the B constants with the Bi form
    '''
    cnt_B = 0

    def replace_C(matched):
        nonlocal cnt_B
        cnt_B += 1
        return 'B{}'.format(cnt_B-1)
    expr_c_sympy = re.sub(r'B', replace_C, expr_c_sympy)
    return expr_c_sympy, cnt_B


def get_expr_C_and_C0(expr, variables, add_bias=True, use_replace_exponent=False):
    """Converts the expr_num string expression
    into an expr_c expression with a constant placeholder
    and the corresponding constant initial value C0
    """
    expr_sympy = sympy.sympify(expr)
    expr_c = to_C_expr(expr_sympy, variables, use_replace_exponent=use_replace_exponent)
    expr_c_sympy = sympy.sympify(expr_c)
    expr_c_sympy, dict_c = densify(expr_c_sympy, variables)
    if add_bias:
        expr_c_sympy = insert_B_on_Add(expr_c_sympy)
    expr_c_sympy_str = str(expr_c_sympy)
    expr_c_sympy_str, cnt_B = replace_B(expr_c_sympy_str)
    for i in range(cnt_B):
        dict_c[sympy.Symbol('B{}'.format(i))] = 0.0
    expr_c_sympy = sympy.sympify(expr_c_sympy_str)
    expr_dense_sympy, dict_final = finallize_const_name(
        expr_c_sympy, dict_c, add_bias=True)
    expr_final_sympy = replace_c_with_a(expr_dense_sympy)
    C0 = np.array(list(dict_final.values()))
    # return str(expr_final_sympy), C0
    return expr_final_sympy, C0          # 


def set_real(expr_c_sympy):
    '''Set all free variables of the expression to real numbers'''
    for var in expr_c_sympy.free_symbols:
        expr_c_sympy = expr_c_sympy.subs(
            var, sympy.Symbol(str(var), real=True))
    return expr_c_sympy


def prun_constant(expr_num_sympy, n_digits=6):
    '''The constants of an expression are 
    rounded by precision and take 
    the value 0 for numbers less than 1e-n in absolute value'''
    epsilon = 10.0**(-n_digits)
    for atom in expr_num_sympy.atoms():
        if isinstance(atom, sympy.core.numbers.Float):
            if abs(atom) < epsilon:
                try:
                    expr_num_sympy = sympy.sympify(
                        expr_num_sympy.subs(atom, sympy.sympify('0')))
                except ZeroDivisionError:
                    expr_num_sympy = expr_num_sympy
            else:
                try:
                    expr_num_sympy = expr_num_sympy.subs(
                        atom, round(atom, n_digits))
                except ZeroDivisionError:
                    expr_num_sympy = expr_num_sympy
                except ValueError:
                    expr_num_sympy = expr_num_sympy

    return expr_num_sympy


def recal_MSE(expr_str, X, Y, variables):
    '''Recalculate the MSE of the expression with numpy'''
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
    try:
        values = {variables[j]: X[:, j] for j in range(X.shape[1])}
        pred = eval(expr_str.lower(), functions, values)
        true = Y[:, 0]
        diff = true - pred
        square = diff ** 2
        return np.mean(square)
    except Exception as e:
        print('recalMSE Error: ',e)
        print('expr_str      : ',expr_str)
        return np.nan
    




class PSRN_Regressor(nn.Module):
    '''
    PSRN Regressor.

    Examples
    ========
    >>> #   input data
    >>> #  ____^______
    >>> # /           |
    >>> # ['x',      'y',   'x+x',        '1.5',      '3']
    >>> # <-n_variables->
    >>> #                 <-n_cross->
    >>> #                            <-trying_const_num->
    >>> # <-----------------n_inputs--------------------->

    '''

    def __init__(self,
                 variables=['x'],
                 operators=['Add', 'Mul', 'Identity',
                            'Sin', 'Exp', 'Neg', 'Inv'],
                 n_symbol_layers=3,
                 n_inputs=5,
                 use_dr_mask=True,
                 dr_mask_dir='./dr_mask',
                 use_const=True,
                 trying_const_num=2,
                 trying_const_range=[0,3],
                 trying_const_n_try=3,
                 
                 device='cuda',
                 ):
        super(PSRN_Regressor, self).__init__()

        if use_const:
            self.trying_const_range = trying_const_range
            self.trying_const_n_try = trying_const_n_try
            self.trying_const_num = trying_const_num
        else:
            if trying_const_num != 0 or trying_const_range is not []:
                print(
                    '[INFO]: setting trying_const_range = [] because use_const = False.')
            self.trying_const_range = []
            self.trying_const_num = 0
            self.trying_const_n_try = trying_const_n_try

        assert n_inputs >= self.trying_const_num + len(variables),\
            'n_inputs error, got {},{},{}'.format(
                n_inputs, self.trying_const_num, len(variables))
        print('[INFO]: Using')
        print('        n_inputs        ', n_inputs)
        print('        trying_const_num', self.trying_const_num)
        print('        len(variables)  ', len(variables))
        self.n_cross = n_inputs - self.trying_const_num - len(variables)
        print('        n_cross         ', self.n_cross)

        self.triu_ls = []
        for i in range(n_inputs):
            self.triu_ls.append(torch.triu_indices(
                i+1, i+1, offset=0, dtype=torch.long, device=device))

        self.N = 1
        self.use_const = use_const
        self.n_inputs = n_inputs
        self.variables = variables
        self.operators = operators
        self.n_symbol_layers = n_symbol_layers
        self.n_variables = len(variables)

        if use_const:
            assert self.trying_const_num > 0, 'If use_const, trying_const_num should > 0'

        forbidden_pattern = r'[cCB]\d*'
        assert not any(
            [re.match(forbidden_pattern, var) for var in variables]
        ), 'you cannot use c, C, or B as variables in regressor'

        self.operators_op = []
        for op_str in operators:
            op = eval(op_str+'_op')
            self.operators_op.append(op())
            
        # load dr mask from dr mask dir.
        self.use_dr_mask = use_dr_mask
        
        if self.use_dr_mask:
            self.dr_mask_dir = dr_mask_dir
            if not os.path.exists(self.dr_mask_dir):
                raise ValueError('dr_mask_dir not exist, got {}'.format(self.dr_mask_dir))
            file_name_mask = f'{self.n_symbol_layers}_{n_inputs}_[{"_".join(self.operators)}]_mask.npy'
            self.dr_mask_path = self.dr_mask_dir + '/' + file_name_mask
            if not os.path.exists(self.dr_mask_path):
                raise ValueError(
                    'dr_mask file not exist, got {}.\nPlease run `{}`'.format(
                    self.dr_mask_path,
                    'python utils/gen_dr_mask.py --n_symbol_layers={} --n_inputs={} --ops="{}"'.format(
                        self.n_symbol_layers,
                        self.n_inputs,
                        str(self.operators).replace(' ',''),
                    )))

            dr_mask = np.load(self.dr_mask_path)
            dr_mask = torch.from_numpy(dr_mask)
            assert dr_mask.dim() == 1, 'dr_mask should be 1-dim, got {}'.format(dr_mask.dim())
        else:
            dr_mask = None
            print('[INFO] use_dr_mask=False. May use more VRAM.')
        
        self.net = PSRN(n_variables=self.n_inputs,
                        operators=operators,
                        n_symbol_layers=n_symbol_layers,
                        dr_mask=dr_mask,
                        device=device)

        self.device = self.net.device
        self.operators_u = []
        self.operators_b = []
        for func in self.net.list[-1].list:
            if func.is_unary:
                self.operators_u.append(func._get_name())
            else:
                self.operators_b.append(func._get_name())
        self.hash_set = set()
        self.value_dict = {}
        self.X = None
        self.Y = None
        self.threshold = None
        self.real_time_display = None
        self.real_time_display_freq = None
        self.real_time_display_ntop = None
        self.prun_const = None
        self.prun_ndigit = None
        self.n_step_simulation = None
        self.probe = None
        self.probe_evalf = None
        self.n_down_sample = None
        self.pareto_frontier = None
        self.use_replace_expo = None
        self.use_strict_pareto = True
    
    def fit(self,
            X,
            Y,
            n_down_sample=200,
            n_step_simulation=100,
            eta=0.99,
            use_threshold=True,
            threshold=1e-10,
            probe=None,
            prun_const=True,
            prun_ndigit=6,
            real_time_display=True,
            real_time_display_freq=20,
            real_time_display_ntop=5,
            ablation_random_MCTS=False,
            dc=0.1,
            add_bias=True,
            together=True,
            top_k=10,
            use_replace_expo=False,
            run_epochs=None,
            use_strict_pareto=True
            ):
        '''fitting data `X (n,m)` and `Y (n,1)` that
        >>> Y = F(X)

        Example
        =======
        >>> flag, pareto_frontier = regressor.fit(X,Y)
        '''

        assert isinstance(
            X, torch.Tensor), 'X must be torch tensor, got {}'.format(type(X))
        assert isinstance(
            Y, torch.Tensor), 'Y must be torch tensor, got {}'.format(type(Y))

        assert X.shape[1] == self.n_variables, 'X.shape[1] must be equal to self.n_variables, got {} and {}'.format(
            X.shape[1], self.n_variables)
        assert X.shape[0] == Y.shape[0], 'X.shape[0] must be equal to Y.shape[0], got {} and {}'.format(
            X.shape[0], Y.shape[0])
        assert Y.shape[1] == 1, 'Y.shape[0] must be equal to 1, got {}'.format(
            Y.shape[1])

        X = X.to(self.net.device)
        Y = Y.to(self.net.device)

        self.X = X
        self.Y = Y
        self.n_down_sample = n_down_sample

        if n_down_sample is None or n_down_sample <= 0 or n_down_sample >= len(X):
            # not using down sample
            # self.X.to(self.net.device)
            # self.Y.to(self.net.device)
            
            self.X_down_sample = self.X
            self.Y_down_sample = self.Y
            print('[INFO]: Down sampling disabled.')
            print('[INFO]: PSRN forwarding will use {}/{} samples.'.format(
                len(self.X_down_sample), len(self.X)
            ))
        else:

            assert n_down_sample <= len(self.X), 'n_down_sample should be less than len(X), got {} and {}'.format(
                n_down_sample, len(self.X))
            
            
            # idx = np.arange(0, X.shape[0], X.shape[0] // n_down_sample)
            # self.X_down_sample = X[idx]
            # self.Y_down_sample = Y[idx]
            
            
            # Calculate the interval for the downsampling
            interval = self.X.shape[0] // n_down_sample
            import random
            # Generate indices for downsampling, taking the offset into account
            idx = (np.arange(n_down_sample) * interval + random.randint(0, X.shape[0] - 1)) % X.shape[0]
            # Downsample X and Y using the generated indices
            self.X_down_sample = X[idx, :]
            self.Y_down_sample = Y[idx, :]
            
            print('[INFO]: Using down sampling,')
            
            print('[INFO]: PSRN forwarding will use {}/{} samples to speed up'.format(
                len(self.X_down_sample), len(self.X)
            ))
            if self.use_const:
                print('[INFO]: Least Square will use all {} samples'.format(
                    len(self.X)
                ))

        self.use_threshold = use_threshold
        if use_threshold:
            print(
                '[INFO]: Using threshold. Algo will stop when MSE < threshold: {}'.format(
                    self.threshold
                ))
        else:
            print(
                '[INFO]: Not using threshold.'
                )
        self.threshold = threshold
        self.real_time_display = real_time_display
        self.real_time_display_freq = real_time_display_freq
        self.real_time_display_ntop = real_time_display_ntop
        self.prun_const = prun_const
        self.prun_ndigit = prun_ndigit
        self.n_step_simulation = n_step_simulation
        self.eta = eta
        self.pareto_frontier = []
        
        self.ablation_random_MCTS = ablation_random_MCTS
        self.dc = dc # constant interval
        self.add_bias = add_bias
        self.together = together
        
        self.top_k = top_k
        self.fitted_expr_c_set = set()
        
        self.use_replace_expo = use_replace_expo
        
        self.use_strict_pareto = use_strict_pareto

        if X.min() >= 0:
            self.is_positive = True
            print('[INFO]: Input is all positive.')
        else:
            self.is_positive = False

        if probe is not None:
            self.probe = prun_constant(
                self.my_simplify(probe, self.together), self.prun_ndigit)
            self.probe_evalf = (self.set_real(sympy.sympify(self.del_float_one(str(self.probe))), self.is_positive)).evalf()
        else:
            self.probe = None

        if real_time_display:
            print('='*60)
            print('[INFO]: Will display best {} expression per {} iterations'.format(
                real_time_display_ntop,
                real_time_display_freq)
            )

        if probe is not None:
            print('='*60)
            print(
                '[INFO]: Using benchmarking mode. Algo will stop when find expression (probe):')
            print('Input --> ', probe)
            print('Sympy --> ', self.probe)
            print('='*60)

        input_expr_ls = self.variables
        n_iter = self.n_step_simulation  # n of MCTS iter in each epoch


        if self.use_const:
            # Constant and linear fitting is performed first
            fitted_c = np.mean(self.Y.cpu().numpy())
            fitted_c_expr = str(fitted_c)
            fitted_c_mse = np.mean((self.Y.cpu().numpy() - fitted_c)**2)
            complexity = 0
            reward = self.get_reward(self.eta, 0, fitted_c_mse)
            expr_reward_mse_complexity_tup_ls = [(fitted_c_expr, reward, fitted_c_mse, complexity)]
            flag = self.pareto_update_and_check(expr_reward_mse_complexity_tup_ls)
            if flag:
                # Stop condition reached
                return True, self.pareto_frontier
            
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(self.X.cpu().numpy(), self.Y.cpu().numpy())
            fitted_linear_expr = str(reg.intercept_[0])
            for i in range(self.n_variables):
                fitted_linear_expr += ' + ' + str(reg.coef_[0][i]) + '*' + str(self.variables[i])
            # print(fitted_linear_expr)
            fitted_linear_expr = str(prun_constant(sympy.sympify(fitted_linear_expr), self.prun_ndigit))
            # print(fitted_linear_expr)
            fitted_linear_mse = np.mean((self.Y.cpu().numpy() - reg.predict(self.X.cpu().numpy()))**2)
            complexity = sympy.count_ops(fitted_linear_expr)
            reward = self.get_reward(self.eta, 1, fitted_linear_mse)
            expr_reward_mse_complexity_tup_ls = [(fitted_linear_expr, reward, fitted_linear_mse, complexity)]
            flag = self.pareto_update_and_check(expr_reward_mse_complexity_tup_ls)
            if flag:
                # Stop condition reached
                return True, self.pareto_frontier

        root_node = MonteCarloNode(input_expr_ls,
                                   self.X_down_sample,
                                   self.operators_op,
                                   0,
                                   self.n_variables + self.n_cross,
                                #    self.trying_const_ls,
                                   self.trying_const_range,
                                   self.trying_const_num,
                                   self.trying_const_n_try,
                                   self)

        global_iter = 0
        current_node = root_node
        n_const_depth = 0
        if self.use_const:
            n_const_depth = 1
        
        if run_epochs is None:
            run_epochs = self.n_cross + n_const_depth
        else:
            assert type(run_epochs) == int and run_epochs <= self.n_cross + n_const_depth
        
        # Make sure to run the same number of steps as before ablation
        for step in range(run_epochs * self.n_step_simulation): 
            
            if self.real_time_display:
                if step % self.real_time_display_freq == 0:
                    self.display_expr_table()
            print('iteration', step, '-'*20)

            # sample a tree depth
            tree_sample_depth_ls = [2, 3]
            tree_sample_operator_op = root_node.operators_op
            
            def get_sampled_tree(depth_ls, operator_op_ls, root_expr_ls, root_data):
                # Choose a random depth from the depth list
                depth = np.random.choice(depth_ls)
                
                # Function to recursively create the expression tree
                def create_expr_tree(current_depth):
                    if current_depth == depth:
                        # If the current depth equals the chosen depth, select a leaf node
                        idx = np.random.choice(len(root_expr_ls))
                        return root_expr_ls[idx], root_data[:, idx]
                    
                    # Choose a random operator from the operator list
                    op = np.random.choice(operator_op_ls)
                    
                    # Recursively create subtrees and apply the operator
                    if op.is_unary:
                        # Unary operation - create one subtree
                        expr, data = create_expr_tree(current_depth + 1)
                        new_expr = op.get_expr(expr)
                        new_data = op.transform_inputs(data)
                    else:
                        # Binary operation - create two subtrees
                        expr1, data1 = create_expr_tree(current_depth + 1)
                        expr2, data2 = create_expr_tree(current_depth + 1)
                        new_expr = op.get_expr(expr1, expr2)
                        new_data = op.transform_inputs(data1, data2)
                        
                    return new_expr, new_data

                # Create the expression tree from the root
                final_expr, final_data = create_expr_tree(0)
                
                # Convert the single expression and its corresponding data into lists
                sampled_new_expr = final_expr
                sampled_new_data = final_data.reshape(-1, 1)  # Convert to 2D array with shape (n, 1)
                
                return sampled_new_expr, sampled_new_data


            # Example usage:
            # sampled_exprs, sampled_data = get_sampled_tree([2, 3], operators, expressions, data)
            
            sampled_new_expr_list = []
            sampled_new_data_catted = root_node.data
            
            
            for _ in range(self.n_cross):
                sampled_new_expr, sampled_new_data = get_sampled_tree(tree_sample_depth_ls,
                                                tree_sample_operator_op,
                                                root_node.expr,
                                                root_node.data
                                                )
            
            
            
                # sampled_new_expr_list = [sampled_new_expr] * 3 # len = m
                # sampled_new_data_catted = np.hstack([sampled_new_data] * 3) # (n, m)
                # sampled_new_data_catted = torch.cat([sampled_new_data] * 3, axis=1) # (n, m)
                
                sampled_new_expr_list.append(sampled_new_expr)
                sampled_new_data_catted = torch.cat([sampled_new_data_catted, sampled_new_data], dim=1)
            
            pointer_node = MonteCarloNode(root_node.expr + sampled_new_expr_list,
                                    #   torch.cat([root_node.data, sampled_new_data_catted], dim=1),
                                    sampled_new_data_catted,
                                      root_node.operators_op,
                                      0, # index
                                      root_node.max_depth,
                                        root_node.trying_const_range,
                                        root_node.trying_const_num,
                                        root_node.trying_const_n_try,
                                        root_node.regressor)

            # pointer_node = current_node
            # while not pointer_node.is_terminal_state():
            #     if not pointer_node.is_visited():
            #         pointer_node.expand()
            #         pointer_node.is_expanded = True
            #         pointer_node.visit = True
            #     else:
            #         # select a random chlid
            #         pointer_node = pointer_node.select(random_select=self.ablation_random_MCTS)

            if self.use_const:
                n_try = self.trying_const_n_try
            else:
                n_try = 1


            e_ls = []
            r_ls = []
            m_ls = []
            c_ls = []
            for try_i in range(n_try):
                
                if self.use_const:
                    bs = self.X_down_sample.shape[0]
                    new_const_ls = np.random.uniform(low=self.trying_const_range[0],
                                                    high=self.trying_const_range[1],
                                                    size=(self.trying_const_num,)).tolist()
                    for i in range(len(new_const_ls)):
                        len_step = self.dc
                        new_const_ls[i] = round(round(new_const_ls[i] / len_step) * len_step + self.dc, 2)
                    n_const = len(new_const_ls)
                    new_data = torch.ones((bs, n_const), device=self.device)
                    for i in range(n_const):
                        new_data[:, i] *= new_const_ls[i]

                    new_expr_ls = [str(num) for num in new_const_ls]
                    
                    pointer_node.expr[-self.trying_const_num:] = new_expr_ls
                    pointer_node.data[:, -self.trying_const_num:] = new_data

                self.net.current_expr_ls = pointer_node.expr
                print('pointer_node.expr', self.net.current_expr_ls)

                expr_best_ls, MSE_min_raw_ls, MSE_mean = self.get_best_expr_and_MSE_topk(
                    pointer_node.data, self.Y_down_sample, self.top_k)
                
                reward_max = -1
                
                for expr_best, MSE_min_raw in zip(expr_best_ls, MSE_min_raw_ls):
                    try:
                        expr_sim = str(self.my_simplify(expr_best, self.together))
                        
                        X = self.X.cpu().numpy()
                        Y = self.Y.cpu().numpy()
                        
                        if not ('nan' in expr_sim or 'oo' in expr_sim):
                            MSE_min_raw = recal_MSE(expr_sim, X, Y, self.variables)
                        if np.isnan(MSE_min_raw) or np.isinf(MSE_min_raw):
                            expr_sim = 'nan'
                        if ('nan' in expr_sim or 'oo' in expr_sim):
                            MSE_min = np.nan
                            reward = 0 
                            complexity = 1e99
                        else:
                            if self.use_const:
                                best_C, MSE_min, expr_c, final_c = self.fit_LS(expr_sim,
                                                                        X,
                                                                        Y,
                                                                        self.variables,
                                                                        MSE_min_raw,
                                                                        add_bias=self.add_bias,
                                                                        together=self.together)
                                if best_C is None:
                                    continue
                                
                                if self.prun_const:
                                    final_c = prun_constant(
                                        sympy.sympify(final_c), self.prun_ndigit)

                                if final_c.is_polynomial():
                                    final_c = final_c.expand()

                                expr_best = str(final_c)
                                
                                
                                print('->',str(expr_sim).ljust(15),
                                    '->',expr_c.ljust(15),
                                    '-> ',str(final_c).ljust(15)
                                    )
                            else:
                                MSE_min = MSE_min_raw
                                expr_best = expr_sim
                                print('expr_best ', expr_best)
                            complexity = sympy.count_ops(expr_best)
                            reward = self.get_reward(self.eta, complexity, MSE_min)

                        if reward > reward_max:
                            reward_max = reward
                        
                        e_ls.append(expr_best)
                        r_ls.append(reward)
                        m_ls.append(MSE_min)
                        c_ls.append(complexity)
                    # except ValueError: Only real AccumulationBounds are supported
                    # except ZeroDivisionError: float division by zero
                    # if there are this error, then skip
                    # except Exception as e:
                    # too general, except only value!
                    except ValueError as e:
                        if 'Only real AccumulationBounds are supported' in str(e) or 'float division by zero' in str(e):
                            continue
                        else:
                            # raise e
                            raise e

                expr_best_ls, reward_ls, mse_ls, complexity_ls = e_ls, r_ls, m_ls, c_ls
                
                for expr_best, reward, mse, complexity in zip(expr_best_ls,
                                                              reward_ls,
                                                              mse_ls,
                                                              complexity_ls):
                    if type(expr_best) is list:
                        for e,r,m,c in zip(expr_best, reward, mse, complexity):
                            expr_reward_mse_complexity_tup_ls = [(
                                e,r,m,c
                            )]
                            # import pandas as pd
                            # df = pd.DataFrame(expr_reward_mse_complexity_tup_ls)
                            # print(df)
                            
                            flag = self.pareto_update_and_check(expr_reward_mse_complexity_tup_ls)
                            
                            if flag:
                                # Stop condition reached
                                return True, self.pareto_frontier 
                    else:
                        expr_reward_mse_complexity_tup_ls = [(
                            expr_best,
                            reward,
                            mse,
                            complexity
                        )]
                        
                        flag = self.pareto_update_and_check(expr_reward_mse_complexity_tup_ls)
                        
                        if flag:
                            # Stop condition reached
                            return True, self.pareto_frontier
                

        return False, self.pareto_frontier

        
    def fit_LS(self, expr_str, X, Y, variables, min_MSE_raw, add_bias, together):
        '''X,Y: (bs,m), (bs,1) numpy'''
        def get_loss_lm(C):

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
            }

            nonlocal expr_c
            expr_c_temp = expr_c

            for i, c in enumerate(C):
                expr_c_temp = expr_c_temp.replace('C{}'.format(i), str(c))

            values = {variables[j]: X[:, j] for j in range(X.shape[1])}
            pred = eval(expr_c_temp.lower(), functions, values)
            true = Y[:, 0]
            diff = true - pred
            square = diff ** 2
            return np.mean(square)

        # Because of the Piecewise problems in the sympy, 
        # a special judgment was made on the sign
        if 'sign' in expr_str or not together:
            expr_num = sympy.sympify(expr_str)
        else:
            expr_num = sympy.simplify(expr_str)

        expr_num = set_real(expr_num)
        if expr_num.is_polynomial():
            expr_num = expr_num.expand()

        expr_c, C0 = get_expr_C_and_C0(expr_num, variables, add_bias=add_bias,
                                       use_replace_exponent=self.use_replace_expo)
        try:
            C0 = np.array(C0).astype(np.float32)
        except:
            return None, np.nan, expr_c, expr_num

        

        # To prevent repeated fitting of formulas of the same form
        pruned_expr_c = prun_constant(expr_c, n_digits=2)
        # print('pruned_expr_c',pruned_expr_c)
        # print('self.fitted_expr_c_set',self.fitted_expr_c_set)
        if pruned_expr_c not in self.fitted_expr_c_set:
            # self.fitted_expr_c_set.add(expr_c) # 
            self.fitted_expr_c_set.add(pruned_expr_c)
        else:
            return None, np.nan, expr_c, expr_num
        
        expr_c = str(expr_c)
        
        try:
            result = opt.minimize(get_loss_lm, C0, method='Powell', tol=1e-6)
            if np.isnan(result.fun):
                raise ValueError
        except:
            return None, np.nan, expr_c, expr_num

        best_C = result.x
        final_c = expr_c
        for i, c in enumerate(best_C):
            final_c = final_c.replace('C{}'.format(i), str(c))

        return result.x, result.fun, expr_c, final_c

    def pareto_update_and_check(self, new_samples):
        index1 = 2 # MSE
        index2 = 3 # Complexity
        
        for sample in new_samples:
            mse = sample[2]
            expr = sample[0]
            if np.isnan(mse) or np.isinf(mse) or 'nan' in expr or 'oo' in expr or 'inf' in expr:
                continue
            if sample[0] in [x[0] for x in self.pareto_frontier]:
                continue
            
            self.pareto_frontier.append(sample)
            
            i = 0
            while i < len(self.pareto_frontier):
                j = i + 1
                while j < len(self.pareto_frontier):
                    if self.pareto_frontier[i][index1] >= self.pareto_frontier[j][index1] and\
                        self.pareto_frontier[i][index2] >= self.pareto_frontier[j][index2]:
                        # The i-th sample is dominated by the j-th sample, so remove it
                        self.pareto_frontier.pop(i)
                        i -= 1
                        break
                    # elif self.pareto_frontier[j][index1] >= self.pareto_frontier[i][index1] and\
                    #     self.pareto_frontier[j][index2] >= self.pareto_frontier[i][index2]:
                    
                    elif (self.use_strict_pareto and self.pareto_frontier[j][index1] >= self.pareto_frontier[i][index1] and\
                        self.pareto_frontier[j][index2] >= self.pareto_frontier[i][index2]) or (not self.use_strict_pareto and self.pareto_frontier[j][index1] > self.pareto_frontier[i][index1] and\
                        self.pareto_frontier[j][index2] > self.pareto_frontier[i][index2]):
                    
                        # The j-th sample is dominated by the i-th sample, so remove it
                        self.pareto_frontier.pop(j)
                        j -= 1
                    j += 1
                i += 1

            # If the new sample is not dominated by any existing sample in the Pareto frontier,
            # apply the check function to it
            if sample in self.pareto_frontier:
                is_terminate = self.pareto_check(sample)
                if is_terminate:
                    return True 
        return False 

    def pareto_check(self, sample):
        expr, reward, mse, complexity = sample
        if self.use_threshold and mse < self.threshold:
            print('='*40)
            print('Algo. stop, because MSE < threshold')
            print('='*40)
            return True
        # elif (self.probe is not None) and self.my_equals(self.my_simplify(expr_best), self.probe):
        elif (self.probe is not None) and self.my_equals(expr, self.probe_evalf):
            print('='*40)
            print('Algo. stop, because expr_best == probe')
            print('MSE',mse)
            print('='*40)
            return True
        else:
            return False
        
    def set_real(self, expr_sympy, positive=False):
        for v in expr_sympy.free_symbols:
            expr_sympy = expr_sympy.subs(v, sympy.Symbol(
                str(v), positive=positive, real=True))
        return expr_sympy

    def my_equals(self, expr, probe_evalf):
        expr = self.set_real(sympy.sympify(self.del_float_one(str(expr))), self.is_positive)
        try:
            is_equal = (expr.evalf()).equals(probe_evalf)
            return is_equal
        except:
            return False

    def del_float_one(self, expr_str):
        '''Replaces '1.0*' with an empty string, but does not match characters such as 31.0* and 1.0**2'''
        import re
        pattern = r'(?<!\d)1\.0\*(?!\*)'
        result = re.sub(pattern, '', expr_str)
        return result

    def my_simplify(self, expr_str, use_together):
        expr_sympy = sympy.sympify(expr_str)
        expr_sympy = set_real(expr_sympy)
        if use_together:
            return sympy.cancel(sympy.together(expr_sympy))
        else:
            return expr_sympy

    def display_expr_table(self, sort_by='reward', descend=None):

        dict_ = {'expr': (0, False), 'mse': (2, False),
                 'reward': (1, True), 'complexity': (3, False)}
        if descend is None:
            descend = dict_[sort_by][1]
        sort_index = dict_[sort_by][0]
        
        pareto_frontier = self.pareto_frontier.copy()
        pareto_frontier.sort(key=lambda x: x[sort_index], reverse=descend)
        
        print('='*73)
        print(
            '|',
            'MSE'.center(10),
            '|',
            'Complexity'.center(10),
            '|',
            'Reward'.center(10),
            '|',
            'Expression'.center(30),
            '|'
        )
        for i, (expr, reward, mse, complexity) in enumerate(pareto_frontier[:self.real_time_display_ntop]):
            print(
                '|',
                format(mse, '10.3e'),
                '|',
                str(complexity).ljust(10),
                '|',
                format(reward, '10.3e'),
                '|',
                expr.ljust(30),
                '|'
            )
        print('='*73)
        return pareto_frontier

    def get_reward(self, eta, complexity, mse):
        return (eta ** complexity) / (1 + math.sqrt(mse))

    def get_best_expr_and_MSE(self, X, Y):

        with torch.no_grad():
            sum_ = torch.zeros((1, self.net.out_dim), device=self.net.device)
            for i in range(X.shape[0]):
                H = self.net.forward(X[i].reshape(1, -1))
                diff = H - Y[i]
                square = diff ** 2
                sum_ += square
            mean = sum_ / X.shape[0]
            mean = mean.reshape(-1)

            # replace all nan, -inf to inf
            mean[torch.isnan(mean)] = float('inf')
            mean[torch.isinf(mean)] = float('inf')

            min_value, min_index = torch.min(mean, dim=0)

            MSE_min = min_value.item()
            MSE_mean = torch.mean(mean).item()

            expr_best = self.net.get_expr(round(min_index.item()))

            return expr_best, MSE_min, MSE_mean

    def get_best_expr_and_MSE_topk(self, X, Y, n_top):
        
        
        self.fitted_expr_c_set = set()

        with torch.no_grad():
            sum_ = torch.zeros((1, self.net.out_dim), device=self.net.device)
            for i in range(X.shape[0]):
                H = self.net.forward(X[i].reshape(1, -1))
                diff = H - Y[i]
                square = diff ** 2
                sum_ += square
            mean = sum_ / X.shape[0]
            mean = mean.reshape(-1)

            # replace all nan, -inf to inf
            mean[torch.isnan(mean)] = float('inf')
            mean[torch.isinf(mean)] = float('inf')

            
            values, indices = torch.topk(mean, n_top, largest=False, sorted=True)
            MSE_min_ls = values.tolist()
            MSE_mean = torch.mean(mean).item()
            expr_best_ls = []
            from tqdm import tqdm
            for i in tqdm(indices.tolist()):
                expr_best_ls.append(self.net.get_expr(round(i)))
            print('expr_best_ls:')
            print('-'*20)
            for expr in expr_best_ls:
                print(expr)
            print('-'*20)
            return expr_best_ls, MSE_min_ls, MSE_mean
        

    def MC(self, node):
        
        select_node = node.select(random_select=self.ablation_random_MCTS)
        if select_node.is_visited() and not select_node.is_terminal_state():
            select_node.expand()
            expr_best_ls, reward_ls, MSE_min_ls, complexity_ls = self.MC(select_node)
        else:
            current_node = select_node
            while not current_node.is_terminal_state():
                while True:
                    is_new_node, new_node = current_node.create_a_random_child()
                    if new_node != None:
                        break
                if is_new_node:
                    new_node.father = current_node
                    current_node.children.append(new_node)
                else:
                    pass
                current_node = new_node
            
            if self.use_const:
                n_try = self.trying_const_n_try
            else:
                n_try = 1
            
            e_ls = []
            r_ls = []
            m_ls = []
            c_ls = []
            for try_i in range(n_try):
                
                if self.use_const:
                    bs = self.X_down_sample.shape[0]
                    new_const_ls = np.random.uniform(low=self.trying_const_range[0],
                                                    high=self.trying_const_range[1],
                                                    size=(self.trying_const_num,)).tolist()
                    for i in range(len(new_const_ls)):
                        len_step = self.dc
                        new_const_ls[i] = round(round(new_const_ls[i] / len_step) * len_step + self.dc, 2)
                    n_const = len(new_const_ls)
                    new_data = torch.ones((bs, n_const), device=self.device)
                    for i in range(n_const):
                        new_data[:, i] *= new_const_ls[i]

                    new_expr_ls = [str(num) for num in new_const_ls]
                    
                    current_node.expr[-self.trying_const_num:] = new_expr_ls
                    current_node.data[:, -self.trying_const_num:] = new_data

                self.net.current_expr_ls = current_node.expr
                print('current_node.expr', self.net.current_expr_ls)

                expr_best_ls, MSE_min_raw_ls, MSE_mean = self.get_best_expr_and_MSE_topk(
                    current_node.data, self.Y_down_sample, self.top_k)
                
                reward_max = -1
                
                for expr_best, MSE_min_raw in zip(expr_best_ls, MSE_min_raw_ls):
                    
                    expr_sim = str(self.my_simplify(expr_best, self.together))
                    
                    X = self.X.cpu().numpy()
                    Y = self.Y.cpu().numpy()
                    
                    if not ('nan' in expr_sim or 'oo' in expr_sim):
                        MSE_min_raw = recal_MSE(expr_sim, X, Y, self.variables)
                    if np.isnan(MSE_min_raw) or np.isinf(MSE_min_raw):
                        expr_sim = 'nan'
                    if ('nan' in expr_sim or 'oo' in expr_sim):
                        MSE_min = np.nan
                        reward = 0 
                        complexity = 1e99
                    else:
                        if self.use_const:
                            best_C, MSE_min, expr_c, final_c = self.fit_LS(expr_sim,
                                                                    X,
                                                                    Y,
                                                                    self.variables,
                                                                    MSE_min_raw,
                                                                    add_bias=self.add_bias,
                                                                    together=self.together)
                            if best_C is None:
                                continue
                            
                            if self.prun_const:
                                final_c = prun_constant(
                                    sympy.sympify(final_c), self.prun_ndigit)

                            if final_c.is_polynomial():
                                final_c = final_c.expand()

                            expr_best = str(final_c)
                            
                            
                            print(str(expr_sim).ljust(15),
                                  '->',expr_c.ljust(15),
                                  '-> ',str(final_c).ljust(15)
                                  )
                        else:
                            MSE_min = MSE_min_raw
                            expr_best = expr_sim
                            print('expr_best ', expr_best)
                        complexity = sympy.count_ops(expr_best)
                        reward = self.get_reward(self.eta, complexity, MSE_min)

                    if reward > reward_max:
                        reward_max = reward
                    
                    e_ls.append(expr_best)
                    r_ls.append(reward)
                    m_ls.append(MSE_min)
                    c_ls.append(complexity)
                
                current_node.backpropagate(reward_max, 1)
            
            return e_ls, r_ls, m_ls, c_ls
            

        # return expr_best, reward, MSE_min, complexity
        return expr_best_ls, reward_ls, MSE_min_ls, complexity_ls


class MonteCarloNode():
    '''MonteCarloNode for PSRN
    '''

    def __init__(self,
                 expr,
                 data,
                 operators_op,
                 index,
                 max_depth,
                #  trying_const_ls,
                #  trying_const_num,
                 trying_const_range,
                 trying_const_num,
                 trying_const_n_try,
                 regressor
                 ):

        self.regressor = regressor

        self.expr = expr
        self.data = data

        self.max_depth = max_depth

        self.n_u = 0
        self.n_b = 0

        self.operators_op = operators_op
        for op in operators_op:
            if op.is_unary:
                self.n_u += 1
            else:
                self.n_b += 1

        self.index = index 

        self.trying_const_range = trying_const_range
        self.trying_const_n_try = trying_const_n_try
        self.trying_const_num = trying_const_num

        self.t = 0
        self.n = 0

        self.n_variable = len(expr)

        self.len_u_block = self.n_variable * self.n_u
        self.len_b_block = self.n_variable * \
            (self.n_variable + 1) // 2 * self.n_b

        if self.next_is_const_child():
            self.n_new_expr = 1
        else:
            self.n_new_expr = self.len_b_block + self.len_u_block

        self.visit = False

        self.children = []
        self.father = None

        self.is_expanded = False

    def is_terminal_state(self):
        if len(self.expr) == self.max_depth + self.trying_const_num:
            return True
        
        else:
            return False

    def next_is_const_child(self):
        if len(self.expr) == self.max_depth:
            return True
        
        else:
            return False

    def is_visited(self):
        return self.visit

    def create_a_random_child(self):
        index = np.random.randint(self.n_new_expr)
        for child in self.children:
            if child.index == index:
                return False, child
        if self.next_is_const_child():
            return True, self.create_a_child_const(index)
        else:
            return True, self.create_a_child(index)

    def create_a_child(self, index):
        if index < self.len_b_block:
            factor = ((self.n_variable) * (self.n_variable + 1) // 2)
            op = self.operators_op[index // factor]

            v_index_1 = self.regressor.triu_ls[self.n_variable -
                                               1][0][index % factor]
            v_index_2 = self.regressor.triu_ls[self.n_variable -
                                               1][1][index % factor]
            expr_1 = self.expr[v_index_1]
            expr_2 = self.expr[v_index_2]
            data_1 = self.data[:, v_index_1]
            data_2 = self.data[:, v_index_2]

            new_expr = op.get_expr(expr_1, expr_2)
            new_data = op.transform_inputs(data_1, data_2).reshape(-1, 1)
        else:

            index -= self.len_b_block

            op = self.operators_op[self.n_b + index // self.n_variable]
            v_index = index % self.n_variable
            expr = self.expr[v_index]
            data = self.data[:, v_index]  # (bs,)

            new_expr = op.get_expr(expr)
            new_data = op.transform_inputs(data).reshape(-1, 1)

        if new_expr in self.expr:
            new_node = None
        else:
            new_node = MonteCarloNode(self.expr + [new_expr],
                                      torch.cat([self.data, new_data], dim=1),
                                      self.operators_op,
                                      index,
                                      self.max_depth,
                                        self.trying_const_range,
                                        self.trying_const_num,
                                        self.trying_const_n_try,
                                        self.regressor)
        return new_node

    def create_a_child_const(self, index):
        bs = self.data.shape[0]
        new_const_ls = np.random.uniform(low=self.trying_const_range[0],
                                         high=self.trying_const_range[1],
                                         size=(self.trying_const_num,)).tolist()
        n_const = len(new_const_ls)
        new_data = torch.ones((bs, n_const), device=self.data.device)
        for i in range(n_const):
            new_data[:, i] *= new_const_ls[i]

        new_expr_ls = [str(num) for num in new_const_ls]
        new_node = MonteCarloNode(self.expr + new_expr_ls,
                                  torch.cat([self.data, new_data], dim=1),
                                  self.operators_op,
                                  index,
                                  self.max_depth,
                                self.trying_const_range,
                                self.trying_const_num,
                                self.trying_const_n_try,
                                self.regressor)
        return new_node

    def backpropagate(self, t_add, n_add):
        self.t += t_add
        self.n += n_add
        self.visit = True
        if not self.father is None:
            self.father.backpropagate(t_add, n_add)
        else:
            self.regressor.N += n_add

    def expand(self):
        for index in range(self.n_new_expr):
            if self.next_is_const_child():
                new_node = self.create_a_child_const(index)
            else:
                new_node = self.create_a_child(index)
            if new_node is None:
                continue
            is_exist = False
            for child in self.children:
                if set(new_node.expr) == set(child.expr):
                    is_exist = True
                    break
                else:
                    pass
            if not is_exist:
                new_node.father = self
                self.children.append(new_node)
        self.is_expanded = True

    def select(self, c=2, random_select=False):
        if random_select:
            return self.children[np.random.randint(len(self.children))]
        else:
            max_ucb_index = -1
            max_ucb_value = -1e20
            for i, child in enumerate(self.children):
                ucb = child.t / (child.n + 1e-6) + c * \
                    math.sqrt(math.log(self.regressor.N) / (child.n + 1e-6))
                if ucb > max_ucb_value:
                    max_ucb_value = ucb
                    max_ucb_index = i
            print('max_ucb_value', max_ucb_value)
            return self.children[max_ucb_index]
    
    
    
