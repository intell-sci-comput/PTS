import sympy as sp
from copy import deepcopy

from ..config import Config
from ..ga.ga import GAPipeline


def remove_abs(expr):
    """
    Removes Abs terms in the expression.
    """
    if expr.is_Atom:
        return expr
    elif expr.func == sp.Abs:
        return expr.args[0]
    else:
        return expr.func(*[remove_abs(arg) for arg in expr.args])

def sqrt_to_exp(expr):
    """
    Converts sqrt terms in the expression to exponentiation form.
    """
    if expr.is_Atom:
        return expr
    elif expr.is_Pow and expr.exp == sp.S(1)/2:
        return sp.Pow(expr.base, 0.5)
    else:
        return expr.func(*[sqrt_to_exp(arg) for arg in expr.args])


def expr_to_tree(expr):
    expr = sqrt_to_exp(expr)
    expr = remove_abs(expr)
    if expr.is_Atom:
        return str(expr)
    else:
        return {
            'op': str(expr.func),
            'args': [expr_to_tree(arg) for arg in expr.args]
        }


def expr_to_tokens_tree(expr):
    now = {}
    if isinstance(expr, str):
        return expr
    expr = deepcopy(expr)
    if expr['op'] == "<class 'sympy.core.mul.Mul'>":
        if len(expr['args'])==1:
            return expr_to_tokens_tree(expr['args'][-1])
        if isinstance(expr['args'][-1], dict) and expr['args'][-1]['op'] == "<class 'sympy.core.power.Pow'>" and \
                expr['args'][-1]['args'][-1] == '-1':
            now['op'] = 'Div'
            right = expr_to_tokens_tree(expr['args'][-1]['args'][0])
            if len(expr['args']) == 2:
                left = expr_to_tokens_tree(expr['args'][0])
            else:
                expr['args'].pop(-1)
                left = expr_to_tokens_tree(expr)
            now['args'] = [left, right]
        else:
            now['op'] = 'Mul'
            right = expr_to_tokens_tree(expr['args'][-1])
            if len(expr['args']) == 2:
                left = expr_to_tokens_tree(expr['args'][0])
            else:
                expr['args'].pop(-1)
                left = expr_to_tokens_tree(expr)
            now['args'] = [left, right]
    elif expr['op'] == "<class 'sympy.core.add.Add'>":
        if isinstance(expr['args'][-1], dict) and expr['args'][-1]['op'] == "<class 'sympy.core.mul.Mul'>" and '-1' in \
                expr['args'][-1]['args']:
            now['op'] = 'Sub'
            expr['args'][-1]['args'].pop(expr['args'][-1]['args'].index("-1"))
            right = expr_to_tokens_tree(expr['args'][-1])
            if len(expr['args']) == 2:
                left = expr_to_tokens_tree(expr['args'][0])
            else:
                expr['args'].pop(-1)
                left = expr_to_tokens_tree(expr)
            now['args'] = [left, right]
        else:
            now['op'] = 'Add'
            right = expr_to_tokens_tree(expr['args'][-1])
            if len(expr['args']) == 2:
                left = expr_to_tokens_tree(expr['args'][0])
            else:
                expr['args'].pop(-1)
                left = expr_to_tokens_tree(expr)
            now['args'] = [left, right]
    elif expr['op'] == "<class 'sympy.core.power.Pow'>":
        if abs(float(expr['args'][-1]) - round(float(expr['args'][-1]))) < 1e-5:
            if float(expr['args'][-1]) < 0:
                now['op'] = 'Div'
                now['args'] = ["1", None]
                if expr['args'][-1] != '-1':
                    now['args'][-1] = expr_to_tokens_tree({
                        'op': "<class 'sympy.core.mul.Mul'>",
                        'args': [expr['args'][0]] * (-int(float(expr['args'][-1])))
                    })
                else:
                    now['args'][-1] = expr_to_tokens_tree(expr['args'][0])
            else:
                if expr['args'][-1] != '1':
                    now = expr_to_tokens_tree({
                        'op': "<class 'sympy.core.mul.Mul'>",
                        'args': [expr['args'][0]] * int(float(expr['args'][-1]))
                    })
                else:
                    now = expr_to_tokens_tree(expr['args'][0])
        elif abs(2 * float(expr['args'][-1]) - round(2 * float(expr['args'][-1]))) < 1e-5:
            expr['args'][-1] = str(int(2 * float(expr['args'][-1])))
            now = {
                "op": "sqrt",
                "args": [expr_to_tokens_tree(expr)]
            }
        else:
            now['op'] = 'Pow'
            now['args'] = [
                expr_to_tokens_tree(exp) for exp in expr['args']
            ]
    else:
        now['op'] = expr['op']
        now['args'] = [
            expr_to_tokens_tree(exp) for exp in expr['args']
        ]
    return now


def tokens_tree_to_expr(tokens):
    if isinstance(tokens, str):
        return tokens
    if tokens['op'] == 'Add':
        return '(' + tokens_tree_to_expr(tokens['args'][0]) + ") + (" + tokens_tree_to_expr(tokens['args'][1]) + ")"
    elif tokens['op'] == 'Sub':
        return '(' + tokens_tree_to_expr(tokens['args'][0]) + ") - (" + tokens_tree_to_expr(tokens['args'][1]) + ")"
    elif tokens['op'] == 'Mul':
        return '(' + tokens_tree_to_expr(tokens['args'][0]) + ") * (" + tokens_tree_to_expr(tokens['args'][1]) + ")"
    elif tokens['op'] == 'Div':
        return '(' + tokens_tree_to_expr(tokens['args'][0]) + ") / (" + tokens_tree_to_expr(tokens['args'][1]) + ")"
    return tokens['op'] + "(" + ', '.join(tokens_tree_to_expr(tk) for tk in tokens['args']) + ")"


def tokens_tree_to_tokens(tokens_tree, new_expr_dict):
    def to_token(s):
        s = s.lower()
        if s in new_expr_dict:
            return new_expr_dict[s]
        return new_expr_dict['c']

    if isinstance(tokens_tree, str):
        return [to_token(tokens_tree)]
    v = [to_token(tokens_tree['op'])]
    assert v[0] != new_expr_dict['c'], f"{tokens_tree['op']} not in expression dict, the new_expr_dict is like this: {new_expr_dict}"
    for tk in tokens_tree['args']:
        v += tokens_tree_to_tokens(tk, new_expr_dict)
    return v


def expression_to_tokens(expr, config):
    expr = expr.replace('Abs(','1.0*(')
    try:
        sympy_expr = sp.sympify(expr).expand()
        expression_tree = expr_to_tree(sympy_expr)

        new_exp = expr_to_tokens_tree(expression_tree)

        new_expr_dict = {
            v.str_name.lower(): k for k, v in config.exp_dict.items()
        }

        tokens = tokens_tree_to_tokens(new_exp, new_expr_dict)

    except Exception:
        sympy_expr = sp.sympify(expr).expand().rewrite('exp')
        expression_tree = expr_to_tree(sympy_expr)
        new_exp = expr_to_tokens_tree(expression_tree)
        new_expr_dict = {
            v.str_name.lower(): k for k, v in config.exp_dict.items()
        }
        tokens = tokens_tree_to_tokens(new_exp, new_expr_dict)

    return tokens

if __name__ == '__main__':
    expression = "x1+cos(x2)**2+sin(log(X3))"
    sympy_expr = sp.sympify(expression).expand()
    expression_tree = expr_to_tree(sympy_expr)
    new_exp = expr_to_tokens_tree(expression_tree)
    config = Config()
    config.init()
    config.set_input(
        x=[[1], [2], [3], [4], [5]],
        t=[11],
        x_=[[1], [2], [3], [4], [5]],
        t_=[11],
    )
    new_expr_dict = {
        v.str_name.lower(): k for k, v in config.exp_dict.items()
    }
    config.verbose = True
    tokens = tokens_tree_to_tokens(new_exp, new_expr_dict)
    ga = GAPipeline(config)
    ga.ga_play([tokens] * 500)
