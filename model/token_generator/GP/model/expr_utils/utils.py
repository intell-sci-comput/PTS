import _thread
import threading
from contextlib import contextmanager
import time
from multiprocessing import Pipe, Process
from typing import Dict

import numpy as np

from .exp_tree_node import Expression


def get_expression(strs: str) -> Expression:
    """
    Get the expression variable for the corresponding expression.
    :param strs: expression string
    :return: the corresponding expression
    """
    exp_dict = {
        "Id": Expression(1, np.array, lambda x: f"{x}"),
        "Add": Expression(2, np.add, lambda x, y: f"({x})+({y})"),
        "Sub": Expression(2, np.subtract, lambda x, y: f"({x})-({y})"),
        "Mul": Expression(2, np.multiply, lambda x, y: f"({x})*({y})"),
        "Div": Expression(2, np.divide, lambda x, y: f"({x})/({y})"),
        "Dec": Expression(1, lambda x: x - 1, lambda x: f"({x})+1"),
        "Pow": Expression(2, np.power, lambda x, y: f"({x})**({y})"),
        "Inc": Expression(1, lambda x: x + 1, lambda x: f"({x})-1"),
        "Neg": Expression(1, np.negative, lambda x: f"-({x})"),
        "Exp": Expression(1, np.exp, lambda x: f"exp({x})"),
        "Log": Expression(1, np.log, lambda x: f"log({x})"),
        "Sin": Expression(1, np.sin, lambda x: f"sin({x})"),
        "Cos": Expression(1, np.cos, lambda x: f"cos({x})"),
        "Asin": Expression(1, np.arcsin, lambda x: f"arcsin({x})"),
        "Atan": Expression(1, np.arctan, lambda x: f"arctan({x})"),
        "Sqrt": Expression(1, np.sqrt, lambda x: f"({x})**0.5"),
        "sqrt": Expression(1, np.sqrt, lambda x: f"({x})**0.5"),
        "N2": Expression(1, np.square, lambda x: f"({x})**2"),
        "Pi": Expression(1, np.pi, lambda x: f"pi*({x})/({x})"),
        "One": Expression(1, 1, lambda x: f"({x})/({x})"),
    }
    if strs in exp_dict:
        temp = exp_dict[strs]
        temp.str_name = strs
        return temp
    return Expression(0, None if strs == 'C' else int(strs[1:]), strs, strs)


def expression_dict(tokens, num_of_var, const) -> Dict[int, Expression]:
    """
    Create expression dictionary key is index, value is Expression class Number of parameters,
     numpy computes expression, string computes expression
    :return: dictionary: key is index, value is Expression
    """

    def generate_expression_dict(expression_list) -> Dict[int, Expression]:
        exp_dict = {}
        for i, expression in enumerate(expression_list):
            exp = get_expression(expression)
            exp.type = i
            exp_dict[i] = exp
            exp.type_name = expression
        return exp_dict

    return generate_expression_dict(
        [f'X{i}' for i in range(1, 1 + num_of_var)] +
        (["C"] if const else []) +
        tokens
    )


class FinishException(Exception):
    """
    Exceptions when getting the correct expression
    """
    pass


@contextmanager
def time_limit(seconds: int, msg=''):
    """
    Timing class Multi-threaded run throws TimeoutError error after seconds
    """
    # if len(threading.enumerate()) >= 20:
    #     for th in threading.enumerate():
    #         if th.name.count('Timer') > 0:
    #             th._stop()
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    try:
        timer.start()
        yield
    except KeyboardInterrupt:
        raise TimeoutError(msg)
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()
