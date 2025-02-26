import os
import numpy as np
import pandas as pd
from utils.data import get_dynamic_data

df, variables_name, target_name = get_dynamic_data('roughpipe','nikuradze')

logf = df['y'].values.reshape(len(df),-1)
logRe = df['l'].values.reshape(len(df),-1)
invRelativeRoughness = df['k'].values.reshape(len(df),-1)

f = 10 ** logf / 100
Re = 10 ** logRe

X = np.log10(Re*np.sqrt(f/32)*(1/invRelativeRoughness))
Y = f ** (-1/2) + 2 * np.log10(1/invRelativeRoughness)

data_df = pd.DataFrame(np.hstack([X,Y]),columns=['X','Y'])

for seed in range(20):

    benchmark_csv_filename = './roughpipe.csv'

    data_df.to_csv(benchmark_csv_filename, header=None,index=False)

    print('running ...',seed)
    os.system("python -m dso.run ./json/NGGP_const_fast_manyops.json --b={} --runs=1 --n_cores_task={} --seed={}".format(
                    benchmark_csv_filename,
                    1,
                    seed
                ))
