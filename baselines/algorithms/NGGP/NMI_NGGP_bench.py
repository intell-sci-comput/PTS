import os
import time
import pandas as pd
import numpy as np
from utils.data import get_benchmark_data

cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)

def append_content_new(txt_path, content_new):
    with open(txt_path,"a",encoding='utf-8') as f:
        f.write(content_new)
        f.close()
    return 

df = pd.read_csv('../../../benchmark/benchmark.csv')
print(df)

benchmark_data_dir = './csv_temp/'
log_time_dir = './log_time/'

if not os.path.exists(benchmark_data_dir):
    os.makedirs(benchmark_data_dir)
if not os.path.exists(log_time_dir):
    os.makedirs(log_time_dir)

for benchmark_name in df['name']:
    
    n_run_each_benchmark = 100
    cnt_run = 0
    for seed in range(n_run_each_benchmark):
        cnt_run += 1
        log_time_path = log_time_dir + benchmark_name + '_{}.txt'.format(seed)
        
        if os.path.exists(log_time_path):
            print('exist ',log_time_path)
            continue
        
        print('running seed {}'.format(seed))
        np.random.seed(seed)
        X, Y, use_constant, expression, variables_name = get_benchmark_data("benchmark.csv",benchmark_name)
        
        df_data = pd.DataFrame(np.hstack([X,Y]))
        
        benchmark_data_dir_this = benchmark_data_dir + benchmark_name + '/'
        if not os.path.exists(benchmark_data_dir_this):
            os.makedirs(benchmark_data_dir_this)
        
        benchmark_csv_filename = benchmark_data_dir_this + '{}_data.csv'.format(seed)
        
        df_data.to_csv(benchmark_csv_filename, index=None, header=None)
        
        time_start = time.time()
        
        if not use_constant:
            os.system("python -m dso.run ./json/NGGP.json --b={} --runs=1 --n_cores_task={} --seed={}".format(
                benchmark_csv_filename,
                1,
                seed
            ))
        else:
            os.system("python -m dso.run ./json/NGGP_const.json --b={} --runs=1 --n_cores_task={} --seed={}".format(
                benchmark_csv_filename,
                1,
                seed
            ))
        
        time_end = time.time()
        time_total = time_end - time_start
        print('='*20)
        print('time_total: ',time_total)
        
        os.remove(benchmark_csv_filename)
        
        append_content_new(log_time_path, str(time_total))
        