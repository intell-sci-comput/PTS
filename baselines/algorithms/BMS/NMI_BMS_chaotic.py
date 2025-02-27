import os
import sys
import time
import warnings
from copy import deepcopy
from optparse import OptionParser
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
import sympy

sys.path.append("../../../Prior")
sys.path.append("../../")
sys.path.append("../../../")

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import BMS.iodata as iodata
from BMS.parallel import Parallel
from dysts.flows import *
from pysindy import SmoothedFiniteDifference
from BMS.utils.sympy_utils import prun_constant
from BMS.utils.data import add_noise

# -----------------------------------------------------------------------------
def read_prior_par(inFileName):
    with open(inFileName) as inf:
        lines = inf.readlines()
    ppar = dict(
        zip(
            lines[0].strip().split()[1:],
            [float(x) for x in lines[-1].strip().split()[1:]],
        )
    )
    return ppar

# -----------------------------------------------------------------------------
def parse_options():
    parser = OptionParser(usage="usage: %prog [options] PPARFILE")
    options = [
        ("-r", "--Drtarget", "Drtarget", 60, "float", "Value of D/r"),
        ("-n", "--nsample", "nsample", 100, "int", "Number of samples"),
        ("-t", "--thin", "thin", 10, "int", "Thinning of the sample"),
        ("-b", "--burnin", "burnin", 500, "int", "Burn-in"),
        ("-T", "--nT", "nT", 10, "int", "Number of temperatures"),
        ("-s", "--Tf", "Tf", 1.2, "float", "Factor between temperatures"),
        ("-a", "--anneal", "anneal", 20, "int", "Annealing threshold"),
        ("-f", "--annealf", "annealf", 5, "float", "Annealing factor"),
    ]
    for short, long, dest, default, type_, help_ in options:
        parser.add_option(short, long, dest=dest, default=default, type=type_, help=help_)
    return parser

class_name_ls = open("dysts_16_flows.txt").readlines()
class_name_ls = [line.strip() for line in class_name_ls]

print(class_name_ls)

sample_size = 1000
noise_level = 0.01

pts_per_period = 100
postprocess = False
n_seeds = 50
N_SEEDS = 20
IS_NOCONST_USE_NP1 = True
IS_USE_ALL_5_CONST = False
TIME_BUDGET = 600
nn, tt, bb = 100, 10, 500

file_name = "BMS_chaotic_results.csv".format(
    nn, tt, bb, TIME_BUDGET
)

############################################### BMS HP
parser = parse_options()
opt, args = parser.parse_args()
prior_path = "./BMS/Prior/"
for seed in range(n_seeds):

    experi_name = "BMS_chaotic_seeds{}".format(N_SEEDS)

    for class_name in class_name_ls:

        dysts_model = eval(class_name)
        dysts_model = dysts_model()
        if (params := {"ForcedBrusselator": ("f", 0), "Duffing": ("gamma", 0), "ForcedVanDerPol": ("a", 0), "ForcedFitzHughNagumo": ("f", 0)}.get(class_name)): setattr(dysts_model, *params)

        np.random.seed(seed)

        print(dysts_model)
        t, data = dysts_model.make_trajectory(
            sample_size,
            return_times=True,
            pts_per_period=pts_per_period,
            resample=True,
            noise=0,
            postprocess=postprocess,
        )

        # add Gaussian noise
        for k in range(data.shape[1]):
            data[:, k : k + 1] = add_noise(data[:, k : k + 1], noise_level, seed)

        t = t.flatten()
        # t: Nx1 ; sol:NxD
        dim = data.shape[1]
        print("dim", dim)
        if dim == 3:
            variables_name = ["x", "y", "z"]
            dotsvarnames = ["xdot", "ydot", "zdot"]
            deriv_idxs = [0, 1, 2]
        elif dim == 4:
            variables_name = ["x", "y", "z", "w"]
            dotsvarnames = ["xdot", "ydot", "zdot", "wdot"]
            deriv_idxs = [0, 1, 2, 3]
        else:
            continue

        sfd = SmoothedFiniteDifference(smoother_kws={"window_length": 5})
        data_deriv = np.zeros((data.shape[0], len(deriv_idxs)))
        for i, idx in enumerate(deriv_idxs):
            print(data[:, idx : idx + 1].shape, t.shape)
            deriv_data_i = sfd._differentiate(data[:, idx : idx + 1], t)
            data_deriv[:, i : i + 1] = deriv_data_i
        data_all = np.hstack([data, data_deriv])

        for idxdot, vardotname in enumerate(dotsvarnames):

            try:
                hp_str = "{}/{}/{}".format(experi_name, class_name, vardotname)
                p = "./log/real_dysts_{}/bms_real/{}/".format(noise_level, hp_str)

                if os.path.exists(p + "pf_{}.csv".format(seed)):
                    print("exist {}, skip.".format(p + "pf_{}.csv".format(seed)))
                    continue
                Input = data
                Output = data_deriv[:, idxdot : idxdot + 1]
                np.random.seed(seed)
                prior_file_list = []
                for root, dirs, files in os.walk(prior_path):
                    for file in files:
                        if file.endswith(".dat"):
                            prior_file_list.append(file)
                X = Input
                Y = Output
                
                use_constant = True
                contains_np = "np5" if use_constant or IS_USE_ALL_5_CONST else "np1"
                n_fea = X.shape[1]
                contains_nv = f"nv{n_fea}"
                print(f"contains_nv {contains_nv} contains_np {contains_np}")

                np_mapping = {"nv3": "np3", "nv4": "np8", "nv5": "np7"}
                contains_np = np_mapping.get(contains_nv, contains_np)

                prefix = f"final_prior_param_sq.named_equations.{contains_nv}.{contains_np}"
                pparfile = next((f"{prior_path}{f}" for f in prior_file_list if f.startswith(prefix)), None)
                print(f"using prior file: {pparfile}")

                with open("progress", "w") as outf:
                    print(f"# OPTIONS  : {opt}\n# ARGUMENTS: {args}", file=outf)

                x, y = pd.DataFrame(X, columns=variables_name), pd.Series(Y.flatten())
                prior_par = read_prior_par(pparfile) if pparfile else ValueError("No prior file found")

                npar = int(pparfile.split(".np")[1].split(".", 1)[0])
                Ts = [1] + [opt.Tf**i for i in range(1, opt.nT)]

                print(x_train_kk := x, y_train_kk := y, sep="\n")
                start = time.time()
                
                p = Parallel(
                    Ts,
                    variables=variables_name,
                    parameters=(
                        ["a%d" % i for i in range(npar)]
                        if (use_constant or IS_NOCONST_USE_NP1)
                        else []
                    ),
                    x=x_train_kk,
                    y=y_train_kk,
                    prior_par=prior_par,
                    bms_probe=None,
                    bms_ispositive=False,
                )

                middle_time = time.time()

                try:
                    xtest = x
                    ypred = p.trace_predict(
                        xtest,
                        samples=opt.nsample,
                        thin=opt.thin,
                        burnin=opt.burnin,
                        anneal=opt.anneal,
                        annealf=opt.annealf,
                        progressfn="progress",
                        reset_files=False,
                        middle_time=middle_time - start,
                        time_budget=TIME_BUDGET,
                    )
                except UnboundLocalError:
                    pass
                end = time.time()
                cost = end - start
                model = deepcopy(p.t1)
                expr = str(model)
                expr_replaced = expr
                for k, v in model.par_values["d0"].items():
                    expr_replaced = expr_replaced.replace(k, f"({v})")
                expr_sympy = sympy.sympify(expr_replaced)

                print(f"time cost: {cost}s\nfinal model: {expr}\nfinal model parameters: {model.par_values['d0']}\n"
                    f"final model with parameters replaced: {expr_replaced}\nfinal model with parameters replaced and sympy: "
                    f"{expr_sympy}\n{expr_sympy.expand()}\n{prun_constant(expr_sympy)}\n"
                    "final checking ...")

                with open(file_name, "a") as f:
                    f.write(f"{class_name},{vardotname},{cost},{expr_replaced},{expr}\n")
 
            except ValueError:
                continue
