from contextlib import contextmanager
import threading
import _thread
from sympy import Symbol, sin, cos, exp, log, count_ops
import sympy as sp


############### evalf timeout exception，useless on Windows ###############
class TimeoutException(Exception):
    def __init__(self, msg=""):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=""):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def has_nested_func(expr):
    if expr is None:
        return False
    if isinstance(expr, str):
        expr = sp.sympify(expr)

    def _has_nested_func(expr):
        if expr.is_Function and expr.func in (sin, cos, exp, log):
            if any(isinstance(arg, (sin, cos, exp, log)) for arg in expr.args):
                return True
        if expr.args:
            return any(_has_nested_func(arg) for arg in expr.args)
        return False

    return _has_nested_func(expr)
