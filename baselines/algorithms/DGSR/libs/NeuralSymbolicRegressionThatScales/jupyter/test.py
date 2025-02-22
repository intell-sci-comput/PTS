# %% [markdown]
# ### Simple example for performing symbolic regression for a set of points

# %%
from nesymres.architectures.model import Model
from nesymres.utils import load_metadata_hdf5
from nesymres.dclasses import FitParams, NNEquation, BFGSParams
from pathlib import Path
from functools import partial
import torch
from sympy import lambdify
import json

import os
cwd = os.getcwd()
print(cwd)

# %%
# Load equation configuration and architecture configuration
import omegaconf
with open('jupyter/100M/eq_setting.json', 'r') as json_file:
    eq_setting = json.load(json_file)

cfg = omegaconf.OmegaConf.load("jupyter/100M/config.yaml")

# %%
# Set up BFGS load rom the hydra config yaml
bfgs = BFGSParams(
    activated=cfg.inference.bfgs.activated,
    n_restarts=cfg.inference.bfgs.n_restarts,
    add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
    normalization_o=cfg.inference.bfgs.normalization_o,
    idx_remove=cfg.inference.bfgs.idx_remove,
    normalization_type=cfg.inference.bfgs.normalization_type,
    stop_time=cfg.inference.bfgs.stop_time,
)


# %%
params_fit = FitParams(word2id=eq_setting["word2id"],
                       id2word={int(k): v for k,
                                v in eq_setting["id2word"].items()},
                       una_ops=eq_setting["una_ops"],
                       bin_ops=eq_setting["bin_ops"],
                       total_variables=list(eq_setting["total_variables"]),
                       total_coefficients=list(
                           eq_setting["total_coefficients"]),
                       rewrite_functions=list(eq_setting["rewrite_functions"]),
                       bfgs=bfgs,
                       # This parameter is a tradeoff between accuracy and fitting time
                       beam_size=cfg.inference.beam_size
                       )

# %%
# weights_path = "../weights/100M.ckpt"
weights_path = "weights/100000_log_-epoch=11-val_loss=0.81.ckpt"

# %%
# Load architecture, set into eval mode, and pass the config parameters
model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
model.eval()
if torch.cuda.is_available():
    model.cuda()

# %%
fitfunc = partial(model.fitfunc, cfg_params=params_fit)

# %%
# Create points from an equation
number_of_points = 500
n_variables = 1

# To get best results make sure that your support inside the max and mix support
max_supp = cfg.dataset_train.fun_support["max"]
min_supp = cfg.dataset_train.fun_support["min"]
X = torch.rand(number_of_points, len(
    list(eq_setting["total_variables"]))) * (max_supp - min_supp) + min_supp
X[:, n_variables:] = 0
target_eq = "x_1*sin(x_1)" #Use x_1,x_2 and x_3 as independent variables
# target_eq = "sin(x_1**2)*cos(x_1)-1"
X_dict = {x: X[:, idx].cpu()
          for idx, x in enumerate(eq_setting["total_variables"])}
y = lambdify(",".join(eq_setting["total_variables"]), target_eq)(**X_dict)

# %%
print("X shape: ", X.shape)
print("y shape: ", y.shape)

# %%
output = fitfunc(X, y)

# %%
print(output)

# %%


# %%


# %%
