import numpy as np

import sympy
import math
from .data import get_dynamic_data, expr_to_Y_pred


def eval_expr_str_on_real_world_dataset(
    dataset_name, dataset_file_name, expr_str, loss_type="MSE"
):
    df, variables_name, target_name = get_dynamic_data(dataset_name, dataset_file_name)
    expr_sympy = sympy.sympify(expr_str)
    X = df[variables_name].values
    Y = df[target_name].values
    Y_pred = expr_to_Y_pred(expr_sympy, X, variables_name)
    if loss_type == "MSE":
        loss = np.mean((Y - Y_pred) ** 2)
    elif loss_type == "MAE":
        loss = np.mean(np.abs(Y - Y_pred))
    elif loss_type == "RMSE":
        loss = np.sqrt(np.mean((Y - Y_pred) ** 2))
    elif loss_type == "NMSE":
        raise ValueError("NMSE not support yet.")
    else:
        raise ValueError("loss_type must be MSE, MAE or RMSE")
    return loss


from .exprutils import time_limit, TimeoutException


def get_sympy_complexity(expr_str):
    complexity_dict = {
        "ADD": 1,
        "SUB": 1,
        "MUL": 1,
        "DIV": 2,
        "POW": 2,
        "SIN": 3,
        "COS": 3,
        "TAN": 3,
        "EXP": 3,
        "LOG": 3,
        "SQRT": 3,
        "NEG": 1,
        "ABS": 4,
        "TANH": 2,
        "SINH": 3,
        "COSH": 2,
        "INV": 2,
        "SIGN": 4,
    }
    try:
        with time_limit(1, "sleep"):
            expr = sympy.sympify(expr_str)
            ops_visual = sympy.count_ops(expr, visual=True)
            ops_visual_str = str(ops_visual)
            complexity = eval(ops_visual_str, complexity_dict)
            return complexity
    except Exception as e:

        return 1e99


def get_reward(eta, complexity, mse):
    return (eta**complexity) / (1 + math.sqrt(mse))


if __name__ == "__main__":
    loss = eval_expr_str_on_real_world_dataset(
        "roughpipe", "nikuradze", "1.41892-0.06674*log(113.57*k**2)", "MSE"
    )
    print(loss)
    loss = eval_expr_str_on_real_world_dataset(
        "roughpipe", "nikuradze", "1.41892-0.06674*log(113.57*k**2)", "MAE"
    )
    print(loss)
