![logo](spl-logo.png)

# SymbolicPhysicsLearner

source code and examples for Symbolic physics learner: Discovering governing equations via Monte Carlo tree search. 

## Dependencies

`requirements.txt`

### Baseline Models
The following packages are not included in the requirements list. Please install at your interest. 

1. base GP model: `pip install gplearn` https://gplearn.readthedocs.io/en/stable/
2. SINDy: `pip install pysindy` https://pysindy.readthedocs.io/en/latest/
3. Deep Symbolic Regression: refer to [dso](https://github.com/brendenpetersen/deep-symbolic-optimization)

## Datasets
### symbolic regression data
To generate training and testing datasets for Nguyen's benchmark problems, run
```
python regression_task/make_datasets.py --task=nguyen-1
```
### dropping ball experiment data 
source data is from https://github.com/briandesilva/discovery-of-physics-from-data

To proces the data into the input of SPL model, run
```
python physics_task/make_datasets_balldrop.py
```
### Lorenz dataset
Lorenz experimental dataset is simulted by MATLAB `ode113` function. 

### double pendulum dataset
Double Pendulum dataset is from camera-recorded experiments provided by https://developer.ibm.com/exchanges/data/all/double-pendulum-chaotic/

Check more details about its pre-processing from `dynamics_task/dp_makedata.ipynb`

## Run Model
### symbolic regression job
Job configurations for Nguyen's benchmark problems are already included. To run experiments with Symbolic Physics Learner, use
```
import sys
import numpy as np
sys.path.append(r'../')
from spl_train import run_spl

output_folder = 'results_dump/' ## directory to save discovered results
save_eqs = True                ## if true, discovered equations are saved to "output_folder" dir

task = 'nguyen-1'
all_eqs, success_rate, all_times = run_spl(task, 
                                           num_run=100, 
                                           transplant_step=10000)
                                           
if save_eqs:
    output_file = open(output_folder + task + '.txt', 'w')
    for eq in all_eqs:
        output_file.write(eq + '\n')
    output_file.close()

print('success rate :', "{:.0%}".format(success_rate))
print('average discovery time is', np.round(np.mean(all_times), 3), 'seconds')                                          
```
To use base-GP model or Deep Symbolic Regression model, please check `run_gp.py` and `run_dsr.py`. 

### physics discovery job
refer to examples in `physics_task/balldrop_tasks.ipynb`

### nonlinear dynamics discovery job
refer to examples in `dynamics_task/spl double pendulum.ipynb` and `dynamics_task/spl lorenz.ipynb`

## Citing the paper
```
@inproceedings{
sun2023symbolic,
title={Symbolic Physics Learner: Discovering governing equations via Monte Carlo tree search},
author={Fangzheng Sun and Yang Liu and Jian-Xun Wang and Hao Sun},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=ZTK3SefE8_Z}
}
```
