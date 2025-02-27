import sympy
import re

def my_simplify(expr_str, use_together):
    expr_sympy = sympy.sympify(expr_str)
    expr_sympy = set_real(expr_sympy)
    if use_together:
        return sympy.cancel(sympy.together(expr_sympy))
    else:
        return expr_sympy

def my_equals(expr, probe_evalf, is_positive):
    expr = set_real(sympy.sympify(del_float_one(str(expr))), is_positive)
    try:
        is_equal = (expr.evalf()).equals(probe_evalf)
        return is_equal
    except:
        return False

def set_real(expr_sympy, positive=False):
    for v in expr_sympy.free_symbols:
        expr_sympy = expr_sympy.subs(
            v, sympy.Symbol(str(v), positive=positive, real=True)
        )
    return expr_sympy

def del_float_one(expr_str):
    
    pattern = r"(?<!\d)1\.0\*(?!\*)"
    result = re.sub(pattern, "", expr_str)
    return result

def prun_constant(expression, n_digits=6):
    zero_threshold = 10.0 ** (-n_digits)
    for atom in expression.atoms():
        if not isinstance(atom, sympy.core.numbers.Float):
            continue
            
        try:
            if abs(atom) < zero_threshold:
                expression = sympy.sympify(
                    expression.subs(atom, sympy.sympify("0"))
                )
            else:
                expression = expression.subs(
                    atom, 
                    round(atom, n_digits)
                )
        except (ZeroDivisionError, ValueError):
            pass
            
    return expression
