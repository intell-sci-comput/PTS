# Standard library imports
import os
import sys
import time
import random
import warnings
import multiprocessing
from optparse import OptionParser
import numpy as np
import pandas as pd
import sympy
from scipy.optimize import OptimizeWarning
from BMS.parallel import Parallel, check_bms_sympy_probe
from BMS.utils.data import get_benchmark_data
from BMS.utils.sympy_utils import (
    my_simplify,
    set_real,
    del_float_one,
    prun_constant
)

warnings.filterwarnings("ignore", category=OptimizeWarning)
sys.path.append("../../../Prior")
sys.path.append("../../")
sys.path.append("../../../")

warnings.filterwarnings("ignore", category=RuntimeWarning)

N_SEEDS = 100
IS_NOCONST_USE_NP1 = True
IS_USE_ALL_5_CONST = True
TIME_BUDGET = 3600
nn, tt, bb = 100, 10, 500
nsample = 1000
thin = 100
burnin = 5000
nT = 10
Tf = 1.2
anneal = 20
annealf = 5

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

def run_single_seed(args):
    csv_name, benchmark_name, seed, file_name, opt, TIME_BUDGET = args
    print(f"Starting seed {seed} for benchmark {benchmark_name}")

    X, Y, use_constant, expression, variables_name = get_benchmark_data(
        csv_name, benchmark_name
    )

    is_positive = X.min() >= 0

    prun_ndigit = 2
    together = False
    probe = prun_constant(my_simplify(expression, together), prun_ndigit)
    probe_evalf = (
        set_real(sympy.sympify(del_float_one(str(probe))), is_positive)
    ).evalf()

    prior_file_list = [f for f in os.listdir("./BMS/Prior/") if f.endswith(".dat")]

    contains_np = "np5" if use_constant or IS_USE_ALL_5_CONST else "np1"
    n_fea = X.shape[1]
    contains_nv = f"nv{n_fea}"

    pparfile = next(
        (
            f"./BMS/Prior/{file}"
            for file in prior_file_list
            if file.startswith(
                f"final_prior_param_sq.named_equations.{contains_nv}.{contains_np}"
            )
        ),
        None,
    )

    prior_par = read_prior_par(pparfile) if pparfile else None

    start = time.time()

    x = pd.DataFrame(X, columns=variables_name)
    y = pd.Series(Y.flatten())

    npar = len(prior_par) if prior_par else 0
    Ts = [1] + [Tf**i for i in range(1, nT)]

    p = Parallel(
        Ts,
        variables=variables_name,
        parameters=(
            ["a%d" % i for i in range(npar)]
            if (use_constant or IS_NOCONST_USE_NP1)
            else []
        ),
        x=x,
        y=y,
        prior_par=prior_par,
        bms_probe=probe_evalf,
        bms_ispositive=is_positive,
    )

    middle_time = time.time()

    try:
        ypred = p.trace_predict(
            x,
            samples=nsample,
            thin=thin,
            burnin=burnin,
            anneal=anneal,
            annealf=annealf,
            progressfn=None,
            reset_files=False,
            middle_time=middle_time - start,
            time_budget=TIME_BUDGET,
        )
    except Exception:
        pass

    end = time.time()

    final_model = p.t1
    expr_best = str(final_model)
    expr_best_replaced = expr_best

    if hasattr(final_model, "par_values") and "d0" in final_model.par_values:
        for k, v in final_model.par_values["d0"].items():
            expr_best_replaced = expr_best_replaced.replace(k, f"({v})")

    bool_ = check_bms_sympy_probe(final_model, probe_evalf, is_positive)

    result = f"{benchmark_name},{bool_},{end-start},{probe_evalf},{expr_best_replaced}\n"

    with open(file_name, "a") as f:
        f.write(result)

    print(f"Finished seed {seed} for benchmark {benchmark_name}")
    return f"Completed: seed {seed} for benchmark {benchmark_name}"


if __name__ == "__main__":
    N_SEEDS = 5
    IS_NOCONST_USE_NP1 = True
    IS_USE_ALL_5_CONST = True
    file_name = "BMS_MainText_SR_results.csv"
    csv_name_ls = ["benchmark.csv", "benchmark_Feynman.csv"]

    # Create a list of all tasks
    all_tasks = []
    for csv_name in csv_name_ls:
        df_all_benchmark = pd.read_csv(f"./benchmark/{csv_name}")
        for benchmark_name in df_all_benchmark["name"]:
            all_tasks.extend(
                [
                    (csv_name, benchmark_name, seed, file_name, None, TIME_BUDGET)
                    for seed in range(N_SEEDS)
                ]
            )
            
    random.shuffle(all_tasks)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as pool:
        for result in pool.imap_unordered(run_single_seed, all_tasks):
            print(result)

    print("All benchmarks and seeds completed.")
