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

prior_path = "./BMS/Prior/"
file_name = 'BMS_EMPS_results.csv'
class_name = 'EMPS'
sys.path.append("../../../BMS/Prior")
sys.path.append("../../")
sys.path.append("../../../")

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from BMS.parallel import Parallel
from dysts.flows import *

from BMS.utils.sympy_utils import prun_constant

from utils.data import get_dynamic_data

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

n_seeds = 50

N_SEEDS = 2
IS_NOCONST_USE_NP1 = True # 可能这个设置，加 大 hp 设置是最有可能找到的
IS_USE_ALL_5_CONST = False

TIME_BUDGET = 90
nn ,tt, bb = 100, 10, 500

parser = parse_options()
opt, args = parser.parse_args()

n_seeds = 20

for seed in range(n_seeds):
    df, variables_name, target_name = get_dynamic_data('emps','emps')
    df = df.iloc[:len(df)//2,:]

    experi_name = '_{}'.format(n_seeds)

    np.random.seed(seed)
    
    Input = df[variables_name].values
    Output = df[target_name].values.reshape(-1,1)

    try:
        prior_file_list = []
        
        for root, dirs, files in os.walk(prior_path):
            for file in files:
                if file.endswith('.dat'):
                    prior_file_list.append(file)
                    
        use_constant = True
        if use_constant or IS_USE_ALL_5_CONST:
            contains_np = 'np5'
        else:
            contains_np = 'np1'
        X = Input
        Y = Output
        n_fea = X.shape[1]

        contains_nv = f"nv{n_fea}"
        print(f"contains_nv {contains_nv} contains_np {contains_np}")

        # 使用字典映射替代多个if判断
        np_mapping = {"nv3": "np3", "nv4": "np8", "nv5": "np7"}
        contains_np = np_mapping.get(contains_nv, contains_np)

        prefix = f"final_prior_param_sq.named_equations.{contains_nv}.{contains_np}"
        pparfile = next((f"{prior_path}{f}" for f in prior_file_list if f.startswith(prefix)), None)
        print(f"using prior file: {pparfile}")

        with open("progress", "w") as outf:
            print(f"# OPTIONS  : {opt}\n# ARGUMENTS: {args}", file=outf)

        x, y = pd.DataFrame(X, columns=variables_name), pd.Series(Y.flatten())
        prior_par = read_prior_par(pparfile) if pparfile else ValueError("No prior file found")

        npar = int(pparfile.split(".np")[1].split(".", 1)[0])  # 更安全的字符串解析
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
            f.write(f"{class_name},{cost},{expr_replaced},{expr}\n")

    except ValueError:
        continue
