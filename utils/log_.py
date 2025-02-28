import pandas as pd
import numpy as np

import time
import os


def save_pareto_frontier_to_csv(save_path, pareto_ls, n_save_top=1000, crit="reward"):
    df_pf = pd.DataFrame(pareto_ls, columns=["expr_str", "reward", "MSE", "complexity"])
    df_pf = df_pf.sort_values(crit, ascending=False)
    df_pf = df_pf.head(n_save_top)
    # t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    df_pf.to_csv(save_path, index=False)
    print("saved pf successfully, n_save_top={}".format(n_save_top))
    print(df_pf)


def create_dir_if_not_exist(path_to_dir):
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)
        print("{} not exist. created.".format(path_to_dir))
        return True
    return False
