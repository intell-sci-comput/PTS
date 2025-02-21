import sys
import os
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn import cross_validation

sys.path.append('../')
from mcmc import *
from parallel import Parallel

# -----------------------------------------------------------------------------
def BIC_SA(x, y, variables, prior_par, npar=None, ns=1000, fn_label='data',
           T_ini=5., T_fin=0.001, T_sched=0.95):
    """Find the formula with the lowest BIC using simulated annealing.
    
    """
    # Initialize
    if npar == None:
        npar = 2 * len(variables)
    t = Tree(
        variables=variables,
        parameters=['a%d' % i for i in range(npar)],
        x=x, y=y,
        prior_par=prior_par,
        BT=T_ini,
    )
    progressFileName = 'bicsa_%s_progress.dat' % fn_label
    traceFileName = 'bicsa_%s_trace.dat' % fn_label
    try:
        os.remove(progressFileName)
    except OSError:
        pass
    try:
        os.remove(traceFileName)
    except OSError:
        pass
    # SA
    while t.BT > T_fin:
        t.get_bic(reset=True, fit=False)
        t.get_energy(bic=False, reset=True)
        print(t.BT, t.bic)
        t.mcmc(burnin=1, thin=1, samples=ns,
               tracefn=traceFileName, progressfn=progressFileName,
               reset_files=False,
               verbose=False)
        t.BT *= T_sched
    # Done
    return t


# -----------------------------------------------------------------------------
def post_SA(x, y, variables, prior_par, npar=None, ns=1000,
           T_ini=5., T_fin=0.001, T_sched=0.95):
    """Find the formula with the largest posterior (lowest energy) using simulated annealing.
    
    """
    # Initialize
    if npar == None:
        npar = 2 * len(variables)
    t = Tree(
        variables=variables,
        parameters=['a%d' % i for i in range(npar)],
        x=x, y=y,
        prior_par=prior_par,
        BT=T_ini,
        PT=min(T_ini, 1.),
    )

    # Anneal
    curT = T_ini
    minE, minF = t.E, deepcopy(t)
    reachedMin = False
    while not reachedMin:
        while t.BT > T_fin:
            # Print the energy 
            print t.BT, t.E, t.bic, t.canonical(), \
                minF.E, minF.bic, minF.canonical(), t.PT
            # MCMC steps at curT
            for s in range(ns):
                t.mcmc_step(verbose=False)
                if t.E < minE:
                    minE = deepcopy(t.E)
                    minF = deepcopy(t)
            # Cool down
            curT *= T_sched
            t.BT = curT
            t.PT = min(curT, 1.)
        # Check if we missed the best visited formula
        if t.E > minE:
            t = deepcopy(minF)
            curT = t.BT = t.PT = .2
        else:
            reachedMin = True
    # Done
    return t

# -----------------------------------------------------------------------------
def model_averaging_valid(x, y, variables, prior_par, npar=None,
                          ns=100, thin=10, fn_label='data',
                          method='kfold', k=2,
                          burnin=5000,
                          start_end=None,
                          parallel=True, par_anneal=100, par_annealf=5.,
                          nT=10, sT=1.20,
                          progressfn='progress.dat'):
    """Validate model averaging using k-fold (method="kfold") or leave-k-out (method="lko").

    """
    if method == 'lko':
        ttsplit = cross_validation.LeavePOut(len(y), k)
    elif method == 'kfold':
        ttsplit = cross_validation.KFold(len(y), shuffle=True, n_folds=k)
    else:
        raise ValueError
    if npar == None:
        npar = 2*len(variables)
    try:
        start, end = start_end
    except:
        start, end = 0, len(ttsplit)

    mse, mae = [], []
    for train_index, test_index in [tt for tt in ttsplit][start: end]:
        xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
        if parallel:
            Ts = [1] + [sT**i for i in range(1, nT)]
            p = Parallel(
                Ts,
                variables=variables,
                parameters=['a%d' % i for i in range(npar)],
                x=xtrain, y=ytrain,
                prior_par=prior_par,
            )
            ypred = p.trace_predict(
                xtest, samples=ns, thin=thin,
                burnin=burnin,
                anneal=par_anneal, annealf=par_annealf,
                progressfn=progressfn, reset_files=False,
            )
        else:
            t = Tree(
                variables=variables,
                parameters=['a%d' % i for i in range(npar)],
                x=xtrain, y=ytrain,
                prior_par=prior_par,
                BT=1.0, PT=1.0,
            )
            ypred = t.trace_predict(xtest, samples=ns, thin=thin,
                                    write_files=False)

        ypredmean = ypred.mean(axis=1)
        ypredmedian = ypred.median(axis=1)

        print ytest
        print ypredmean
        print ypredmedian

        mse.append(np.mean((ytest - ypredmean)**2))
        mae.append(np.mean(np.abs(ytest - ypredmedian)))

    return mse, mae
        
