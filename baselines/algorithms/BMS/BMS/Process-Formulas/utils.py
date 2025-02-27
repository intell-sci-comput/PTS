import re
import signal
import os
import errno
from functools import wraps

class TimeoutError(Exception):
    pass


### CLASSES
class BanlistedExpressionException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
class FunctionLikeExpression(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
class LatexSymbolNotFound(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
class SymbolRequiresArg(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
class TimeoutError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
class UnwantedOperator(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


### FUNCTIONS

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            return result

        return wraps(func)(wrapper)

    return decorator

def get_bounded_term(s, pos=0, d1='(', d2=')', reverse = False):
    count = 0
    rv = 0
    op = 1
    exp = s[pos:]
    if reverse:
        exp = reversed(s[:pos+1])
        op = -1

    for c in exp:
        if c == d1: count += 1
        elif c == d2: count -=1
        if count == 0: break
        rv += op

    return pos + rv + 1

def list_to_regexp_or(l, escape = True, add_head = False, add_trail = True):
    x = l[0]
    trail = head = ""
    if add_trail: trail = "(?![a-zA-Z])"
    if add_head: head = "(?<![a-zA-Z])"
    if escape: x = re.escape(x)

    if re.search("[a-zA-Z]+", x):
        ret = head+x+trail
    else:
        ret = x

    for x in l[1:]:
        if escape: x = re.escape(x)
        if re.search("[a-zA-Z]+", x):
            ret = ret + "|" + head + x + trail
        else:
            ret = ret + "|" + x
    return ret

def go_untouchable(start, end, key, symbol_type="var", force_replace=False):
    num_token_1 = num_token_2 = (end-start)/2
    if not (end-start)%2 : num_token_2 -= 1
    if symbol_type == "var":
        return "|"+"@"*(num_token_1-1) + str(key) + "@"*(num_token_2-1)+"|"
    elif symbol_type == "const":
        return "?"*num_token_1 + str(key) + "?"*num_token_2
    elif symbol_type == "argop":
        return "|"+"#"*(num_token_1-1) + str(key) + "#"*(num_token_2-1)+"|"
    elif symbol_type == "op":
        if force_replace:
            return "|"+"#"*(num_token_1-1) + str(key) + "#"*(num_token_2-1)+"|"
        else:
            return " "*num_token_1 + str(key) + " "*num_token_2
    else:
        return " "*num_token_1 + str(key) + " "*num_token_2


def check_undefined_symbols(exp):
    symbols = list_to_regexp_or(LATEX_NOARG_SYMBOLS.keys()) + list_to_regexp_or(LATEX_ARG_SYMBOLS.keys()) + list_to_regexp_or(OTHER_SYMBOLS)
    for m in re.finditer("\\\\[a-zA-Z]+", exp):
        if m.group() not in symbols:
            raise LatexSymbolNotFound(m.group())


def check_banned_expressions(exp):
    for w in BANLISTED_TOKENS:
        if w in exp:
            raise BanlistedExpressionException(w)

    for w in BANLISTED_WORDS:
        m = re.search("[^a-zA-Z]%s" % w, exp)
        if m:
            raise BanlistedExpressionException(w)


def check_derived_expressions(exp):
    m = re.search("[ (]*(d ?([a-zA-Z]|\|@*[0-9]@*\||\^[a-zA-Z0-9])[ )]*/[( ]*d ?([a-zA-Z]|\|@*[0-9]@*\||\^[a-zA-Z0-9]))[ )]*", exp)
    if m:
        raise BanlistedExpressionException(m.group())

    m = re.search("_{([a-z])\\1{1,}}", exp)
    if m:
        raise BanlistedExpressionException(m.group())

    m = re.search("u_[xyztr]", exp)
    if m:
        raise BanlistedExpressionException(m.group())


def prepare_function_candidates(exp, function_candidates):
    for m in re.finditer("(([a-zA-Z]+)(_{?[\\\\a-zA-Z0-9]+}?)?(\^{?[\\\\()\-+*/a-zA-Z0-9]+}?)?)\(", exp):
        if exp[m.start()-1] == '\\':
            symbol = "\\"+m.group(2)
            if (symbol in LATEX_NOARG_SYMBOLS and LATEX_NOARG_SYMBOLS[symbol]["type"] == 'var'):
                pass #LATEX_NOARG_SYMBOLS type var can behave like functions
            else:
                continue #If not var, we are not in a function

        end_term = get_bounded_term(exp, m.end()-1)

        #Check function chars like , ; |
        for t in FUNCTION_TERMS:
            if t in exp[m.end()-1:end_term]:
                function_candidates.append(m.group())

        if len(exp)>end_term and exp[end_term] == '^':
            continue

        x = re.search("[^a-zA-Z0-9*^\\\\]", exp[m.end():end_term-1])
        if not x:
            function_candidates.append(m.group())



def check_functions(exp, function_candidates):
    for m in re.finditer("(([a-zA-Z]+)(_{?[\\\\a-zA-Z0-9]+}?)?(\^{?[\\\\()\-+*/a-zA-Z0-9]+}?)?)\(", exp):
        end_term = get_bounded_term(exp, m.end()-1)

        if m.group() in function_candidates:
            raise FunctionLikeExpression(exp[m.start():end_term])


### GLOBAL VARS
BANLISTED_TOKENS = [ 'dt', 'dx', 'dy', 'dz', 'df', 'dX', 'dY', 'df', '\'', '|_', '||' ]
BANLISTED_WORDS = [ 'for', 'then', 'if' ]
FUNCTION_TERMS = [ ',', ';', '|' ] #For example, f(a+b) is not function, place + here and it will be treated as function.
SPLIT_OPS = [ '=', '<', '>', '<=', '>=', '\\le', '\\ge', '\\geq', '\\leq', '\\approx', '\\sim', '\\equiv' ]
NO_SPLIT_WORDS_SYMBOLS = [ '*', '\\times', '\\cdot' ] #Don't split words when this symbols appear in the expression
SPECIAL_NUMBERS = [ 'Ar', 'Be', 'Bi', 'Br', 'Ca', 'Da', 'De', 'Ec', 'Ek', 'Eo', 'Eu', 'Fr', 'Ga', 'Gz', 'Gr', 'Ha', 'La', 'Su', 'Pr', 'Mg', 'Nu', 'Oh', 'Pe', 'Ra', 'Re', 'Ro', 'Sc', 'Sh', 'St', 'We', 'Wi',  ]


SYMPY_SPECIAL_SYMBOLS = {

'N'  : { 'op': 'N_n', 'regexp': '(?:^|[^a-zA-Z])(N)(?:[^a-zA-Z_]|$)' },  #Function
'S'  : { 'op': 'S_s', 'regexp': '(?:^|[^a-zA-Z])(S)(?:[^a-zA-Z_]|$)' },  #Singleton class
'Q'  : { 'op': 'Q_q', 'regexp': '(?:^|[^a-zA-Z])(Q)(?:[^a-zA-Z_]|$)' },  #Function

}

SPECIAL_FUNCTIONS = {

'\\Gamma' : { 'op': 'Gamma' },

}

LATEX_NOARG_SYMBOLS = {

'\\sigma'   : { 'op': 'sigma', 'type': 'var' },
'\\alpha'   : { 'op': 'alpha', 'type': 'var' },
'\\omega'   : { 'op': 'omega', 'type': 'var' },
'\\beta'    : { 'op': 'beta', 'type': 'var' },
'\\theta'   : { 'op': 'theta', 'type': 'var' },
'\\delta'   : { 'op': 'delta', 'type': 'var' },
'\\ell'     : { 'op': 'ell', 'type': 'var' },
'\\gamma'   : { 'op': 'gamma', 'type': 'var' },
'\\phi'     : { 'op': 'phi', 'type': 'var' },
'\\chi'     : { 'op': 'chi', 'type': 'var' },
'\\tau'     : { 'op': 'tau', 'type': 'var' },
'\\lambda'  : { 'op': 'lambda', 'type': 'var' },
'\\hbar'    : { 'op': 'hbar', 'type': 'var' },
'\\psi'     : { 'op': 'psi', 'type': 'var' },
'\\eta'     : { 'op': 'eta', 'type': 'var' },
'\\rho'     : { 'op': 'rho', 'type': 'var' },
'\\nu'      : { 'op': 'nu', 'type': 'var' },
'\\epsilon' : { 'op': 'epsilon', 'type': 'var' },
'\\mu'      : { 'op': 'mu', 'type': 'var' },
'\\xi'      : { 'op': 'xi', 'type': 'var' },
'\\varphi'  : { 'op': 'varphi', 'type': 'var' },
'\\zeta'    : { 'op': 'zeta', 'type': 'var' },
'\\eta'     : { 'op': 'eta', 'type': 'var' },
'\\kappa'   : { 'op': 'kappa', 'type': 'var' },
'\\varphi'  : { 'op': 'varphi', 'type': 'var' },
'\\varepsilon' : { 'op': 'varepsilon', 'type': 'var' },

'\\Omega'   : { 'op': 'Omega', 'type': 'var' },
'\\Phi'     : { 'op': 'Phi', 'type': 'var' },
'\\Psi'     : { 'op': 'Psi', 'type': 'var' },
'\\Gamma'   : { 'op': 'Gamma', 'type': 'var' },
'\\Theta'   : { 'op': 'Theta', 'type': 'var' },
'\\Beta'    : { 'op': 'Beta', 'type': 'var' },
'\\Lambda'  : { 'op': 'Lambda', 'type': 'var' },

'\\pi'      : { 'op': 'pi', 'type': 'const' },

'\\left'    : { 'op': '(', 'type': 'op', 'replace': False },
'\\('       : { 'op': '(', 'type': 'op', 'replace': False },
'\\right'   : { 'op': ')', 'type': 'op', 'replace': False },
'\\)'       : { 'op': ')', 'type': 'op', 'replace': False },
'\\,'       : { 'op': ' ', 'type': 'op', 'replace': False },
'\\!'       : { 'op': ' ', 'type': 'op', 'replace': False },
'\\times'   : { 'op': '*', 'type': 'op', 'replace': False },
'\\cdot'    : { 'op': '*', 'type': 'op', 'replace': False },
'\\over'    : { 'op': ')/(', 'type': 'op', 'replace': False }, # Replace { a*b +c \over a+b } with ( a*b +c )/( a+b )
# '\\gcd'     : { 'op': 'gcd', 'type': 'op', 'replace': True }, #gcd op in latex has no params so we trust the user uses it correctly. Disabled coz sympy always transforms it to 1 with symbols

}

LATEX_ARG_SYMBOLS = {

# '\\dot'     : { 'op': 'dot', 'extra_param': False },

'\\sin'     : { 'op': 'sin', 'extra_param': False },
'\\sinh'    : { 'op': 'sinh', 'extra_param': False },
'\\cos'     : { 'op': 'cos', 'extra_param': False },
'\\cosh'    : { 'op': 'cosh', 'extra_param': False },
'\\arccos'  : { 'op': 'arccos', 'extra_param': False },
'\\arcsin'  : { 'op': 'arcsin', 'extra_param': False },
'\\arctan'  : { 'op': 'arctan', 'extra_param': False },
'\\tan'     : { 'op': 'tan', 'extra_param': False },
'\\tanh'    : { 'op': 'tanh', 'extra_param': False },
'\\cot'     : { 'op': 'cot', 'extra_param': False },
'\\coth'    : { 'op': 'coth', 'extra_param': False },
'\\sec'     : { 'op': 'sec', 'extra_param': False },
'\\sech'    : { 'op': 'sech', 'extra_param': False },

'\\sqrt'    : { 'op': 'sqrt', 'extra_param': True, 'alt_op': 'root' },
'\\log'     : { 'op': 'log', 'extra_param': True },
'\\ln'      : { 'op': 'log', 'extra_param': True },
'\\exp'     : { 'op': 'exp', 'extra_param': False },

'\\frac'    : { 'op': '/', 'extra_param': False},
'\\tfrac'   : { 'op': '/', 'extra_param': False},
'\\cfrac'   : { 'op': '/', 'extra_param': False},
'\\binom'   : { 'op': 'binomial', 'extra_param': False},

}

SYMPY_OP_STRING = {
"<class 'sympy.core.mul.Mul'>" : "Mul",
"<class 'sympy.core.add.Add'>" : "Add",
"<class 'sympy.core.power.Pow'>" : "Pow",
}


#Not used to get data, just to have them registered as LATEX symbols in order to don't ban them coz they are not found.
OTHER_SYMBOLS = [ '\\text' ]
