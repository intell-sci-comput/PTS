import sys
import pandas as pd

sys.path.append('../../20-Fit/')
from mcmc import *

sys.path.append('../')
from data_analysis import BIC_SA

import iodata

VARS = [
    'CDH1',
    'CDH2',
    'CDH3',
    'ALPHACAT',
    'BETACAT',
    'P120',
    'ZO1',
]
Y = 'Sxx'
NS = 1000

if __name__ == '__main__':
    # Read the data
    inFileName = 'cadhesome_protein.csv'
    data, x, y = iodata.read_data(
        ylabel=Y, xlabels=VARS, in_fname=inFileName,
    )

    prior_par = read_prior_par('../../10-Prior/prior_param.named_equations.nv7.np14.2015-10-29 11:18:27.834512.dat')
    print prior_par

    # SA
    t = BIC_SA(x, y, VARS, prior_par, ns=2000, fn_label=inFileName,
               T_ini=5., T_fin=0.001, T_sched=0.95)
