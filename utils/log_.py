import pandas as pd
import numpy as np

import time
import os

def save_pareto_frontier_to_csv(save_path, pareto_ls, n_save_top=1000, crit='reward'):
    df_pf = pd.DataFrame(pareto_ls, columns=['expr_str', 'reward', 'MSE', 'complexity'])
    df_pf = df_pf.sort_values(crit,ascending=False)
    df_pf = df_pf.head(n_save_top)
    # t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    df_pf.to_csv(save_path, index=False)
    print('saved pf successfully, n_save_top={}'.format(n_save_top))
    print(df_pf)

def create_dir_if_not_exist(path_to_dir):
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)
        print('{} not exist. created.'.format(path_to_dir))
        return True
    return False


def get_other_info(path_of_other_dir,replace_dict={'varold':'varnew'}):
    import os
    p = path_of_other_dir
    files = os.listdir(p)
    for filename in files:
        if filename.endswith('pf.csv'):
            pf_name = filename
        if filename.endswith('.csv') and ('pf' not in filename) and ('hof' not in filename):
            time_name = filename
    import pandas as pd
    df_pf = pd.read_csv(os.path.join(p,pf_name))
    expressions = [e[1:-1] for e in df_pf['expression'].tolist()]
    # replace_dict = {'x1':'x',
    #                 'x2':'y',
    #                 'x3':'z'}
    for k,v in replace_dict.items():
        expressions = [e.replace(k,v) for e in expressions]
    
    df_time = pd.read_csv(os.path.join(p,time_name))
    time_cost = df_time['time'].iloc[-1]
    
    return expressions, time_cost