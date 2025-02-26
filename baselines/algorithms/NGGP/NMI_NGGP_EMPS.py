import os 
import numpy as np
import sympy as sp
from utils.data import get_dynamic_data

df, variables_name, target_name = get_dynamic_data('emps','emps')

benchmark_csv_filename = './emps.csv'
seed = 0

df.to_csv(benchmark_csv_filename, header=None,index=False)

os.system("python -m dso.run ./json/NGGP_const_fast_manyops.json --b={} --runs=1 --n_cores_task={} --seed={}".format(
                benchmark_csv_filename,
                1,
                seed
            ))