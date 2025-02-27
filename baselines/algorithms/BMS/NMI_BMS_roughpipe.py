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

import BMS.iodata as iodata
from BMS.parallel import Parallel
from dysts.flows import *

from BMS.utils.sympy_utils import prun_constant

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

if __name__ == '__main__':
    
    N_SEEDS = 20
    TIME_BUDGET = 90

    file_name = 'BMS_roughpipe_results.csv'

    with open(file_name,'a') as f:
        pass 
    
    for seed in range(N_SEEDS):
        parser = parse_options()
        opt, args = parser.parse_args()
        dset = 'RoughPipes'
        VARS = iodata.XVARS[dset]
        Y = iodata.YLABS[dset]
        pparfile = prior_path+'final_prior_param_sq.named_equations.nv1.np5.2017-10-18 18_07_35.227360.dat'
        
        Drtarget = opt.Drtarget
        
        inFileName = './data/%s' % (iodata.FNAMES[dset])
        data, x, y = iodata.read_data(
            dset, ylabel=Y, xlabels=VARS, in_fname=inFileName,
        )

        progressfn = 'progress'
        with open(progressfn, 'w') as outf:
            print('# OPTIONS  :', opt, file=outf)
            print('# ARGUMENTS:', args, file=outf)
        
        l = x.values[:,0:1]
        k = x.values[:,1:2]
        y = y.values.reshape(-1,1)
        f = (10 ** y / 100)
        Re = 10 ** l
        x = np.log10(Re*np.sqrt(f/32)*(1/k))
        y = f ** (-1/2) + 2 * np.log10(1/k)

        x = pd.DataFrame(x,columns=['x'])
        y = pd.Series(y.flatten())
        
        if pparfile != None:
            prior_par = read_prior_par(pparfile)
        npar = pparfile[pparfile.find('.np') + 3:]
        npar = int(npar[:npar.find('.')])
        Ts = [1] + [opt.Tf**i for i in range(1, opt.nT)]
        
        x_train_kk = x
        y_train_kk = y
        print(x_train_kk)
        print(y_train_kk)
        
        start = time.time()
        
        p = Parallel(
            Ts,
            variables=['x'],
            parameters=['a%d' % i for i in range(npar)],
            x=x_train_kk, y=y_train_kk,
            prior_par=prior_par,
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