import numpy as np

import sympy
import math
from utils.data import get_dynamic_data, expr_to_Y_pred

def eval_expr_str_on_real_world_dataset(dataset_name, dataset_file_name, expr_str, loss_type='MSE'):
    df, variables_name, target_name = get_dynamic_data(dataset_name,dataset_file_name)
    expr_sympy = sympy.sympify(expr_str)
    X = df[variables_name].values
    Y = df[target_name].values
    Y_pred = expr_to_Y_pred(expr_sympy, X, variables_name)
    if loss_type=='MSE':
        loss = np.mean((Y - Y_pred) ** 2)
    elif loss_type=='MAE':
        loss = np.mean(np.abs(Y - Y_pred))
    elif loss_type=='RMSE':
        loss = np.sqrt(np.mean((Y - Y_pred) ** 2))
    elif loss_type=='NMSE':
        raise ValueError('NMSE not support yet.')
    else:
        raise ValueError('loss_type must be MSE, MAE or RMSE')
    return loss




def get_reward(eta, complexity, mse):
    return (eta ** complexity) / (1 + math.sqrt(mse))

if __name__ == '__main__':
    loss = eval_expr_str_on_real_world_dataset('roughpipe','nikuradze','1.41892-0.06674*log(113.57*k**2)','MSE')
    print(loss)
    loss = eval_expr_str_on_real_world_dataset('roughpipe','nikuradze','1.41892-0.06674*log(113.57*k**2)','MAE')
    print(loss)