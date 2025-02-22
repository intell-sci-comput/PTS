import numpy as np
from numba import jit

CONSTANT = 'constant'

ARITY_OPERATORS = {
    # Elementary functions
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "pow": 2,
    "inv": 1,
    "pow2": 1,
    "pow3": 1,
    "pow4": 1,
    "pow5": 1,
    "sqrt": 1,
    "exp": 1,
    "ln": 1,
    "abs": 1,

    # Trigonometric Functions
    "sin": 1,
    "cos": 1,
    "tan": 1,

    # Trigonometric Inverses
    "asin": 1,
    "acos": 1,
    "atan": 1,

    # Hyperbolic Functions
    "sinh": 1,
    "cosh": 1,
    "tanh": 1,
    "coth": 1,

    # Additional Functions
    "log": 1,
    "inv": 1,
    "neg": 1
}

RECURSION_LIMIT = 300


def compute_arities(eq):
    if len(eq) > RECURSION_LIMIT:
        raise ValueError('Recursion Limit Hit')
    arities = []
    arity = 1
    for t in eq:
        if t in ARITY_OPERATORS:
            arity += ARITY_OPERATORS[t] - 1
        else:
            arity -= 1
        arities.append(arity)
    return np.array(arities)


def eq_remove_negative_integer_powers(eq):
    max_len = len(eq)
    if max_len > RECURSION_LIMIT:
        raise ValueError('Recursion Limit Hit')
    i = 0
    while i < max_len:
        if eq[i] == 'pow':
            arities = compute_arities(eq[i:])
            exponent_end_idx = arities.tolist().index(0)
            if eq[exponent_end_idx + i].startswith('-') and eq[exponent_end_idx + i][1:].isdigit():
                eq[exponent_end_idx +
                    i] = str(abs(int(eq[exponent_end_idx + i])))
                eq.insert(i, 'inv')
                max_len = len(eq)
        i += 1
        if i > RECURSION_LIMIT:
            raise ValueError('Recursion Limit Hit')
    return eq


def eq_remap_pow(eq):
    max_len = len(eq)
    if max_len > RECURSION_LIMIT:
        raise ValueError('Recursion Limit Hit')
    i = 0
    while i < max_len:
        if eq[i] == 'pow':
            arities = compute_arities(eq[i:])
            exponent_end_idx = arities.tolist().index(0)
            if eq[exponent_end_idx + i].isdigit():
                operator = int(eq[exponent_end_idx + i])
                var = ','.join(eq[i + 1:exponent_end_idx + i])
                rep = f'mul,{var},' * (operator - 1) + var
                to_replace = f'pow,{var},{operator}'
                eq = ','.join(eq).replace(to_replace, rep).split(',')
                max_len = len(eq)
        i += 1
        if i > RECURSION_LIMIT:
            raise ValueError('Recursion Limit Hit')
    return eq


def eq_remap_negative_constants(eq):
    max_len = len(eq)
    if max_len > RECURSION_LIMIT:
        raise ValueError('Recursion Limit Hit')
    i = 0
    while i < max_len:
        if eq[i] == 'add':
            arities = compute_arities(eq[i:])
            possible_integer = arities.tolist().index(0)
            if eq[possible_integer + i].startswith('-') and eq[possible_integer + i][1:].isdigit():
                eq[possible_integer +
                    i] = f'{abs(int(eq[possible_integer + i])):.1f}'
                eq.insert(i + possible_integer, 'neg')
                max_len = len(eq)
                continue
            possible_integer = arities.tolist().index(1)
            if eq[possible_integer + i].startswith('-') and eq[possible_integer + i][1:].isdigit():
                eq[possible_integer +
                    i] = f'{abs(int(eq[possible_integer + i])):.1f}'
                eq.insert(i + possible_integer, 'neg')
                max_len = len(eq)
                continue
        i += 1
        if i > RECURSION_LIMIT:
            raise ValueError('Recursion Limit Hit')
    return eq


def eq_constant_mul(eq):
    max_len = len(eq)
    if max_len > RECURSION_LIMIT:
        raise ValueError('Recursion Limit Hit')
    i = 0
    while i < max_len:
        if eq[i] == 'mul':
            arities = compute_arities(eq[i:])
            first_leaf_idx = arities.tolist().index(1)
            second_leaf_idx = arities.tolist().index(0)
            if eq[i + first_leaf_idx].isdigit():
                operator = int(eq[first_leaf_idx + i])
                var = ','.join(
                    eq[first_leaf_idx + i + 1:second_leaf_idx + i + 1])
                rep = f'add,{var},' * (operator - 1) + var
                to_replace = f'mul,{operator},{var}'
                eq = ','.join(eq).replace(to_replace, rep).split(',')
                max_len = len(eq)
            elif (eq[i + first_leaf_idx].startswith('-') and eq[i + first_leaf_idx][1:].isdigit()):
                operator = abs(int(eq[first_leaf_idx + i]))
                var = ','.join(
                    eq[first_leaf_idx + i + 1:second_leaf_idx + i + 1])
                varc = 'neg,' + var
                rep = f'add,{varc},' * (operator - 1) + varc
                to_replace = f'mul,-{operator},{var}'
                eq = ','.join(eq).replace(to_replace, rep).split(',')
                max_len = len(eq)
            elif eq[i + second_leaf_idx].isdigit() and (first_leaf_idx + 1) == second_leaf_idx:
                operator = int(eq[second_leaf_idx + i])
                var = ','.join(eq[i + 1:second_leaf_idx + i])
                rep = f'add,{var},' * (operator - 1) + var
                to_replace = f'mul,{var},{operator}'
                eq = ','.join(eq).replace(to_replace, rep).split(',')
                max_len = len(eq)
            elif (eq[i + second_leaf_idx].startswith('-') and eq[i + second_leaf_idx][1:].isdigit()) and (first_leaf_idx + 1) == second_leaf_idx:
                operator = abs(int(eq[second_leaf_idx + i]))
                var = ','.join(eq[i + 1:second_leaf_idx + i])
                varc = 'neg,' + var
                rep = f'add,{varc},' * (operator - 1) + varc
                to_replace = f'mul,{var},-{operator}'
                eq = ','.join(eq).replace(to_replace, rep).split(',')
                max_len = len(eq)
        if i == max_len:
            break
        i += 1
        if i > RECURSION_LIMIT:
            raise ValueError('Recursion Limit Hit')
    return eq

def eq_remove_float_constants(eq):
    max_len = len(eq)
    if max_len > RECURSION_LIMIT:
        raise ValueError('Recursion Limit Hit')
    i = 0
    while i < max_len:
        if eq[i] == 'mul':
            arities = compute_arities(eq[i:])
            first_leaf_idx = arities.tolist().index(1)
            second_leaf_idx = arities.tolist().index(0)
            if eq[i + first_leaf_idx].isdigit(): # or (eq[i + first_leaf_idx].startswith('-' and eq[i + first_leaf_idx][1:].isdigit())): # Handle this case like above
                
                
                eq = ','.join(eq).replace('mul,constant,','').split(',')
                max_len = len(eq)
            elif eq[i + second_leaf_idx] == CONSTANT:
                var = ','.join(eq[i + 1:second_leaf_idx + i])
                rep = f'mul,{var},{CONSTANT}'
                eq = ','.join(eq).replace(rep,var).split(',')
                max_len = len(eq)
        if i == max_len:
            break
        i += 1
        if i > RECURSION_LIMIT:
            raise ValueError('Recursion Limit Hit')
    return eq

def eq_remove_constants(eq):
    max_len = len(eq)
    if max_len > RECURSION_LIMIT:
        raise ValueError('Recursion Limit Hit')
    # Process Adds
    i = 0
    while i < max_len:
        if eq[i] == 'add':
            arities = compute_arities(eq[i:])
            first_leaf_idx = arities.tolist().index(1)
            second_leaf_idx = arities.tolist().index(0)
            if eq[i + first_leaf_idx] == CONSTANT:
                eq = ','.join(eq).replace('add,constant,','').split(',')
                max_len = len(eq)
            elif eq[i + second_leaf_idx] == CONSTANT and eq[i + first_leaf_idx + 1] == CONSTANT:
                var = ','.join(eq[i + 1:second_leaf_idx + i])
                rep = f'add,{var},{CONSTANT}'
                eq = ','.join(eq).replace(rep,var).split(',')
                max_len = len(eq)
        if i == max_len:
            break
        i += 1
        if i > RECURSION_LIMIT:
            raise ValueError('Recursion Limit Hit')
    # Process Muls
    i = 0
    while i < max_len:
        if eq[i] == 'mul':
            arities = compute_arities(eq[i:])
            first_leaf_idx = arities.tolist().index(1)
            second_leaf_idx = arities.tolist().index(0)
            if eq[i + first_leaf_idx] == CONSTANT:
                eq = ','.join(eq).replace('mul,constant,','').split(',')
                max_len = len(eq)
            elif eq[i + second_leaf_idx] == CONSTANT and eq[i + first_leaf_idx + 1] == CONSTANT:
                var = ','.join(eq[i + 1:second_leaf_idx + i])
                rep = f'mul,{var},{CONSTANT}'
                eq = ','.join(eq).replace(rep,var).split(',')
                max_len = len(eq)
        if i == max_len:
            break
        i += 1
        if i > RECURSION_LIMIT:
            raise ValueError('Recursion Limit Hit')
    es = []
    for e in eq:
        if e.isdigit():
            es.append(f'{int(e):.1f}')
        elif e.startswith('-') and e[1:].isdigit():
            es.append(f'{int(e):.1f}')
        else:
            es.append(e)
    return es

def eq_sympy_prefix_to_token_library(eq_sympy_prefix,
                                     log=True,
                                     inv=True,
                                     const=True,
                                     remap_vars=True,
                                     neg=True,
                                     constant_mul=True,
                                     remap_pow=True,
                                     remap_negative_constants=True,
                                     nesymres_map=True):
    str_ = ','.join(eq_sympy_prefix)
    if neg:
        str_ = str_.replace('mul,-1,', 'neg,')
    # if sub:
    #     str_ = str_.replace('add,mul,-1', 'sub')
    if log:
        str_ = str_.replace('ln,', 'log,')
    if const:
        str_ = str_.replace('c,', 'const,')
    if nesymres_map:
        str_ = str_.replace('acos,', 'arccos,')
        str_ = str_.replace('asin,', 'arcsin,')
        str_ = str_.replace('atan,', 'arctan,')
    eq = str_.split(',')
    if inv:
        eq = eq_remove_negative_integer_powers(eq)
    if remap_pow:
        eq = eq_remap_pow(eq)
    if constant_mul:
        eq = eq_constant_mul(eq)
    if remap_negative_constants:
        eq = eq_remap_negative_constants(eq)
    if remap_vars:
        eq = [j.replace('_', '') if j[0] == 'x' else j for j in eq]
    return eq


def replace_with_div(eq):
    max_len = len(eq)
    if max_len > RECURSION_LIMIT:
        raise ValueError('Recursion Limit Hit')
    i = 0
    while i < max_len:
        if eq[i] == 'mul':
            arities = compute_arities(eq[i:])
            first_leaf_idx = arities.tolist().index(1)
            second_leaf_idx = arities.tolist().index(0)
            if eq[i + first_leaf_idx + 1] == 'inv':
                del eq[i + first_leaf_idx + 1]
                eq[i] = 'div'
                max_len = len(eq)
            elif eq[i + second_leaf_idx -1] == 'inv':
                del eq[i + second_leaf_idx -1]
                eq[i] = 'div'
                max_len = len(eq)
        if i == max_len:
            break
        i += 1
        if i > RECURSION_LIMIT:
            raise ValueError('Recursion Limit Hit')
    return eq

def replace_with_neg_with_sub(eq):
    max_len = len(eq)
    if max_len > RECURSION_LIMIT:
        raise ValueError('Recursion Limit Hit')
    i = 0
    while i < max_len:
        if eq[i] == 'add':
            arities = compute_arities(eq[i:])
            first_leaf_idx = arities.tolist().index(1)
            second_leaf_idx = arities.tolist().index(0)
            if eq[i + first_leaf_idx + 1] == 'neg':
                del eq[i + first_leaf_idx + 1]
                eq[i] = 'sub'
                max_len = len(eq)
            elif eq[i + second_leaf_idx -1] == 'neg':
                del eq[i + second_leaf_idx -1]
                eq[i] = 'sub'
                max_len = len(eq)
        if i == max_len:
            break
        i += 1
        if i > RECURSION_LIMIT:
            raise ValueError('Recursion Limit Hit')
    return eq

def test_eq_remap_negative_constants():
    assert eq_remap_negative_constants(
        ['mul', 'x1', 'add', 'log', 'x1', 'sin', '1']) == ['mul', 'x1', 'add', 'log', 'x1', 'sin', '1']
    assert eq_remap_negative_constants(
        ['add', '-1', 'mul', 'cos', 'x1', 'sin', 'mul', 'x1', 'x1']) == ['add', 'neg', '1.0', 'mul', 'cos', 'x1', 'sin', 'mul', 'x1', 'x1']
    assert eq_remap_negative_constants(
        ['add', 'mul', 'cos', 'x1', 'sin', 'mul', 'x1', 'x1', '-1']) == ['add', 'mul', 'cos', 'x1', 'sin', 'mul', 'x1', 'x1', 'neg', '1.0']
    assert eq_remap_negative_constants(
        ['add', '-1', 'add', '-2', 'cos', 'x1']) == ['add', 'neg', '1.0', 'add', 'neg', '2.0', 'cos', 'x1']



def test_eq_remove_negative_integer_powers():
    assert eq_remove_negative_integer_powers(
        ['pow', 'x_2', '-2']) == ['inv', 'pow', 'x_2', '2']
    assert eq_remove_negative_integer_powers(
        ['pow', 'cos', 'pow', 'x_2', '2', '-2']) == ['inv', 'pow', 'cos', 'pow', 'x_2', '2', '2']
    assert eq_remove_negative_integer_powers(
        ['pow', 'cos', 'pow', 'x_2', '-2', '-2']) == ['inv', 'pow', 'cos', 'inv', 'pow', 'x_2', '2', '2']
    assert eq_remove_negative_integer_powers(
        ['mul', 'x_1', 'pow', 'x_2', '-2']) == ['mul', 'x_1', 'inv', 'pow', 'x_2', '2']
    assert eq_remove_negative_integer_powers(
        ['pow', 'cos', 'x_2', '-2']) == ['inv', 'pow', 'cos', 'x_2', '2']



def test_remap_pow():
    assert eq_remap_pow(['pow', 'x_1', '2']) == ['mul', 'x_1', 'x_1']
    assert eq_remap_pow(['pow', 'x_1', '3']) == [
        'mul', 'x_1', 'mul', 'x_1', 'x_1']
    assert eq_remap_pow(['pow', 'cos', 'x_1', '3']) == [
        'mul', 'cos', 'x_1', 'mul', 'cos', 'x_1', 'cos', 'x_1']
    assert eq_remap_pow(['inv', 'pow', 'x_2', '2']) == [
        'inv', 'mul', 'x_2', 'x_2']
    assert eq_remap_pow(['inv', 'pow', 'cos', 'inv', 'pow', 'x_2', '2', '2']) == [
        'inv', 'mul', 'cos', 'inv', 'mul', 'x_2', 'x_2', 'cos', 'inv', 'mul', 'x_2', 'x_2']
    assert eq_remap_pow(['inv', 'pow', 'cos', 'pow', 'x_2', '2', '2']) == [
        'inv', 'mul', 'cos', 'mul', 'x_2', 'x_2', 'cos', 'mul', 'x_2', 'x_2']
    assert eq_remap_pow(['inv', 'pow', 'cos', 'pow', 'x_2', '3', '2']) == [
        'inv', 'mul', 'cos', 'mul', 'x_2', 'mul', 'x_2', 'x_2', 'cos', 'mul', 'x_2', 'mul', 'x_2', 'x_2']



def test_eq_constant_mul():
    assert eq_constant_mul(['mul', 'x1', 'add', 'log', 'x1', 'sin', '1']) == [
        'mul', 'x1', 'add', 'log', 'x1', 'sin', '1']
    assert eq_constant_mul(['mul', '2', 'x_1']) == ['add', 'x_1', 'x_1']
    assert eq_constant_mul(['mul', '3', 'x_1']) == [
        'add', 'x_1', 'add', 'x_1', 'x_1']
    assert eq_constant_mul(['mul', '4', 'x_1']) == [
        'add', 'x_1', 'add', 'x_1', 'add', 'x_1', 'x_1']
    assert eq_constant_mul(['mul', '-2', 'x_1']
                           ) == ['add', 'neg', 'x_1', 'neg', 'x_1']
    assert eq_constant_mul(['mul', 'x_1', '2']) == ['add', 'x_1', 'x_1']
    assert eq_constant_mul(['mul', 'x_1', '-2']
                           ) == ['add', 'neg', 'x_1', 'neg', 'x_1']
    assert eq_constant_mul(['mul', '2', 'cos', 'x_1']) == [
        'add', 'cos', 'x_1', 'cos', 'x_1']
    assert eq_constant_mul(['mul', 'cos', 'x_1', '3']) == [
        'add', 'cos', 'x_1', 'add', 'cos', 'x_1', 'cos', 'x_1']
    assert eq_constant_mul(['mul', 'sin', 'mul', '2', 'x_1', '3']) == [
        'add', 'sin', 'add', 'x_1', 'x_1', 'add', 'sin', 'add', 'x_1', 'x_1', 'sin', 'add', 'x_1', 'x_1']
    assert eq_constant_mul(['mul', '3', 'sin', 'mul', '2', 'x_1']) == [
        'add', 'sin', 'add', 'x_1', 'x_1', 'add', 'sin', 'add', 'x_1', 'x_1', 'sin', 'add', 'x_1', 'x_1']
    assert eq_constant_mul(['mul', '3', 'sin', 'mul', 'x_1', '2']) == [
        'add', 'sin', 'add', 'x_1', 'x_1', 'add', 'sin', 'add', 'x_1', 'x_1', 'sin', 'add', 'x_1', 'x_1']



def test_eq_sympy_prefix_to_token_library():
    assert eq_sympy_prefix_to_token_library(
        ['mul', 'x1', 'add', 'log', 'x1', 'sin', '1']) == ['mul', 'x1', 'add', 'log', 'x1', 'sin', '1']
    assert eq_sympy_prefix_to_token_library(
        ['pow', 'x_2', '-2']) == ['inv', 'mul', 'x2', 'x2']
    assert eq_sympy_prefix_to_token_library(
        ['pow', 'cos', 'pow', 'x_2', '2', '-2']) == ['inv', 'mul', 'cos', 'mul', 'x2', 'x2', 'cos', 'mul', 'x2', 'x2']
    assert eq_sympy_prefix_to_token_library(
        ['pow', 'cos', 'pow', 'x_2', '-2', '-2']) == ['inv', 'mul', 'cos', 'inv', 'mul', 'x2', 'x2', 'cos', 'inv', 'mul', 'x2', 'x2']
    assert eq_sympy_prefix_to_token_library(
        ['mul', 'x_1', 'pow', 'x_2', '-2']) == ['mul', 'x1', 'inv', 'mul', 'x2', 'x2']
    assert eq_sympy_prefix_to_token_library(
        ['pow', 'cos', 'x_2', '-2']) == ['inv', 'mul', 'cos', 'x2', 'cos', 'x2']
    assert eq_sympy_prefix_to_token_library(['pow', 'x_1', '2']) == [
        'mul', 'x1', 'x1']
    assert eq_sympy_prefix_to_token_library(['pow', 'x_1', '3']) == [
        'mul', 'x1', 'mul', 'x1', 'x1']
    assert eq_sympy_prefix_to_token_library(['pow', 'cos', 'x_1', '3']) == [
        'mul', 'cos', 'x1', 'mul', 'cos', 'x1', 'cos', 'x1']
    assert eq_sympy_prefix_to_token_library(['inv', 'pow', 'x_2', '2']) == [
        'inv', 'mul', 'x2', 'x2']
    assert eq_sympy_prefix_to_token_library(['inv', 'pow', 'cos', 'inv', 'pow', 'x_2', '2', '2']) == [
        'inv', 'mul', 'cos', 'inv', 'mul', 'x2', 'x2', 'cos', 'inv', 'mul', 'x2', 'x2']
    assert eq_sympy_prefix_to_token_library(['inv', 'pow', 'cos', 'pow', 'x_2', '2', '2']) == [
        'inv', 'mul', 'cos', 'mul', 'x2', 'x2', 'cos', 'mul', 'x2', 'x2']
    assert eq_sympy_prefix_to_token_library(['inv', 'pow', 'cos', 'pow', 'x_2', '3', '2']) == [
        'inv', 'mul', 'cos', 'mul', 'x2', 'mul', 'x2', 'x2', 'cos', 'mul', 'x2', 'mul', 'x2', 'x2']
    assert eq_sympy_prefix_to_token_library(['mul', '2', 'x_1']) == [
        'add', 'x1', 'x1']
    assert eq_sympy_prefix_to_token_library(['mul', '3', 'x_1']) == [
        'add', 'x1', 'add', 'x1', 'x1']
    assert eq_sympy_prefix_to_token_library(['mul', '4', 'x_1']) == [
        'add', 'x1', 'add', 'x1', 'add', 'x1', 'x1']
    assert eq_sympy_prefix_to_token_library(['mul', '-2', 'x_1']
                                            ) == ['add', 'neg', 'x1', 'neg', 'x1']
    assert eq_sympy_prefix_to_token_library(['mul', 'x_1', '2']) == [
        'add', 'x1', 'x1']
    assert eq_sympy_prefix_to_token_library(['mul', 'x_1', '-2']
                                            ) == ['add', 'neg', 'x1', 'neg', 'x1']
    assert eq_sympy_prefix_to_token_library(['mul', '2', 'cos', 'x_1']) == [
        'add', 'cos', 'x1', 'cos', 'x1']
    assert eq_sympy_prefix_to_token_library(['mul', 'cos', 'x_1', '3']) == [
        'add', 'cos', 'x1', 'add', 'cos', 'x1', 'cos', 'x1']
    assert eq_sympy_prefix_to_token_library(['mul', 'sin', 'mul', '2', 'x_1', '3']) == [
        'add', 'sin', 'add', 'x1', 'x1', 'add', 'sin', 'add', 'x1', 'x1', 'sin', 'add', 'x1', 'x1']
    assert eq_sympy_prefix_to_token_library(['mul', '3', 'sin', 'mul', '2', 'x_1']) == [
        'add', 'sin', 'add', 'x1', 'x1', 'add', 'sin', 'add', 'x1', 'x1', 'sin', 'add', 'x1', 'x1']
    assert eq_sympy_prefix_to_token_library(['mul', '3', 'sin', 'mul', 'x_1', '2']) == [
        'add', 'sin', 'add', 'x1', 'x1', 'add', 'sin', 'add', 'x1', 'x1', 'sin', 'add', 'x1', 'x1']

    # Problematic expressions
    assert eq_sympy_prefix_to_token_library(
        ['add', 'pow', 'sin', 'pow', 'x_1', '3', '5', 'mul', '-1', 'pow', 'x_2', '4']) == ['add', 'mul', 'sin', 'mul', 'x1', 'mul', 'x1', 'x1', 'mul', 'sin', 'mul', 'x1', 'mul', 'x1', 'x1', 'mul', 'sin', 'mul', 'x1', 'mul', 'x1', 'x1', 'mul', 'sin', 'mul', 'x1', 'mul', 'x1', 'x1', 'sin', 'mul', 'x1', 'mul', 'x1', 'x1', 'neg', 'mul', 'x2', 'mul', 'x2', 'mul', 'x2', 'x2']


def test_eq_remove_constants():
    assert eq_remove_constants(['add', 'x1', 'mul', 'constant', 'mul', 'add', '1', 'mul', 'x1', 'constant', 'mul', 'add', '1', 'mul', 'x1', 'constant', 'add', '1', 'mul', 'x1', 'constant']) == ['add', 'x1', 'mul', 'add', '1.0', 'x1', 'mul', 'add', '1.0', 'x1', 'add', '1.0', 'x1']
    assert eq_remove_constants(['mul', 'x1', 'inv', 'tan', 'add', 'x1', 'constant']) == ['mul', 'x1', 'inv', 'tan', 'x1']
    assert eq_remove_constants(['mul', 'x1', 'inv', 'tan', 'add', 'x1', 'neg', 'mul', 'add', 'x1', 'constant', 'add', 'x1', 'constant']) == ['mul', 'x1', 'inv', 'tan', 'add', 'x1', 'neg', 'mul', 'x1', 'x1']
    assert eq_remove_constants(['mul', 'x1', 'inv', 'tan', 'mul', 'constant', 'mul', 'add', 'neg', '1.0', 'mul', 'x1', 'constant', 'add', 'neg', '1.0', 'mul', 'x1', 'constant']) == ['mul', 'x1', 'inv', 'tan', 'mul', 'add', 'neg', '1.0', 'x1', 'add', 'neg', '1.0', 'x1']
    assert eq_remove_constants(['add', '1', '-5']) == ['add', '1.0', '-5.0']
    assert eq_remove_constants(['add', '0', '4']) == ['add', '0.0', '4.0']
    # assert eq_remove_constants(['add', 'x1', 'mul', 'add', '1', 'x1', 'mul', 'add', '1', 'x1', 'add', '1', 'x1']) == ['add', 'x1', 'mul', 'x1', 'mul', 'x1', 'x1']
    # assert eq_remove_constants(['mul', 'x1', 'exp', 'sin', 'mul', 'x1', 'constant']) == ['mul', 'x1', 'exp', 'sin', 'x1'] # Invalid input

def test_replace_with_div():
    assert replace_with_div(['mul', 'x1', 'inv', 'x2']) == ['div', 'x1', 'x2']
    assert replace_with_div(['mul', 'x1', 'inv', 'add', 'x1', 'x2']) == ['div', 'x1', 'add', 'x1', 'x2']
    assert replace_with_div(['mul', 'add', 'x1', 'x2', 'inv', 'x3']) == ['div', 'add', 'x1', 'x2', 'x3']

def test_replace_neg_with_sub():
    assert replace_with_neg_with_sub(['div', 'x1', 'mul', 'x4', 'mul', 'x5', 'add', 'x3', 'neg', 'x2']) == ['div', 'x1', 'mul', 'x4', 'mul', 'x5', 'sub', 'x3', 'x2']


if __name__ == "__main__":
    test_eq_remove_negative_integer_powers()
    test_remap_pow()
    test_eq_constant_mul()
    test_eq_remap_negative_constants()
    test_eq_sympy_prefix_to_token_library()
    test_eq_remove_constants()
    test_replace_with_div()
    test_replace_neg_with_sub()
    print('Tests passed')
