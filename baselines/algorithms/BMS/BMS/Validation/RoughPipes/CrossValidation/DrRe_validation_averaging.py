import sys
import numpy as np
import pandas as pd
from optparse import OptionParser

sys.path.append('../../../Prior')
#from fit_prior import read_prior_par
sys.path.append('../../')
sys.path.append('../../../')
from parallel import Parallel
from data_analysis import model_averaging_valid

import iodata

# -----------------------------------------------------------------------------
def read_prior_par(inFileName):
    with open(inFileName) as inf:
        lines = inf.readlines()
    ppar = dict(zip(lines[0].strip().split()[1:],
                    [float(x) for x in lines[-1].strip().split()[1:]]))
    return ppar

# -----------------------------------------------------------------------------
def parse_options():
    """Parse command-line arguments.

    """
    parser = OptionParser(usage='usage: %prog [options] PPARFILE')
    parser.add_option("-D", "--Drtarget", dest="Drtarget", default=60,
                      type='float',
                      help="Value of D/r (default: 60)")
    parser.add_option("-R", "--Retarget", dest="Retarget", default=5.5,
                      type='float',
                      help="Upper bound of Log(Re) used for training (default: 5.5)")
    parser.add_option("-n", "--nsample", dest="nsample", default=1000,
                      type='int',
                      help="Number of samples (default: 1000)")
    parser.add_option("-t", "--thin", dest="thin", default=100,
                      type='int',
                      help="Thinning of the sample (default: 100)")
    parser.add_option("-b", "--burnin", dest="burnin", default=5000,
                      type='int',
                      help="Burn-in (default: 5000)")
    parser.add_option("-T", "--nT", dest="nT", default=10,
                      type='int',
                      help="Number of temperatures (default: 10)")
    parser.add_option("-s", "--Tf", dest="Tf", default=1.2,
                      type='float',
                      help="Factor between temperatures (default: 1.20)")
    parser.add_option("-a", "--anneal", dest="anneal", default=20,
                      type='int',
                      help="Annealing threshold. If there are no tree swaps for more than this number of steps, the parallel tempering is annealed (default: 20)")
    parser.add_option("-f", "--annealf", dest="annealf", default=5,
                      type='float',
                      help="Annealing factor: all temperatures are multiplied by this factor during the heating phase of the annealing (default: 5)")
    return parser

if __name__ == '__main__':
    # Arguments
    parser = parse_options()
    opt, args = parser.parse_args()
    dset = 'RoughPipes'
    VARS = iodata.XVARS[dset]
    Y = iodata.YLABS[dset]
    pparfile = args[0]
    Drtarget = opt.Drtarget
    Retarget = opt.Retarget
    
    # Read the data
    inFileName = '../data/%s' % (iodata.FNAMES[dset])
    data, x, y = iodata.read_data(
        dset, ylabel=Y, xlabels=VARS, in_fname=inFileName,
    )

    # Prepare output files
    progressfn = 'DrRe%g_%g_validation_averaging.progress' % (Drtarget, Retarget)
    with open(progressfn, 'w') as outf:
        print >> outf, '# OPTIONS  :', opt
        print >> outf, '# ARGUMENTS:', args
    
    # Create a validation set with points with fixed Drratio and a maximum Re
    xtrain = x[(np.abs(x['Drratio'] - Drtarget) > 1.e-5) & 
               (x['LogRe'] < Retarget)]
    xtest = x[(np.abs(x['Drratio'] - Drtarget) <= 1.e-5) |
              (x['LogRe'] >= Retarget)]
    ytrain = y[(np.abs(x['Drratio'] - Drtarget) > 1.e-5) &
               (x['LogRe'] < Retarget)]
    ytest = y[(np.abs(x['Drratio'] - Drtarget) <= 1.e-5) | 
              (x['LogRe'] >= Retarget)]
    print(xtest, '\n', ytest)

    # Make the predictions by averaging
    if pparfile != None:
        prior_par = read_prior_par(pparfile)
    npar = pparfile[pparfile.find('.np') + 3:]
    npar = int(npar[:npar.find('.')])
    Ts = [1] + [opt.Tf**i for i in range(1, opt.nT)]
    p = Parallel(
        Ts,
        variables=VARS,
        parameters=['a%d' % i for i in range(npar)],
        x=xtrain, y=ytrain,
        prior_par=prior_par,
    )
    ypred = p.trace_predict(
        xtest, samples=opt.nsample, thin=opt.thin,
        burnin=opt.burnin,
        anneal=opt.anneal, annealf=opt.annealf,
        progressfn=progressfn,
        reset_files=False,
    )
    
    ypredmean = ypred.mean(axis=1)
    ypredmedian = ypred.median(axis=1)

    # Output
    xtrain.to_csv('DrRe%g_%g_validation_averaging.xtrain.csv' % (Drtarget, Retarget))
    xtest.to_csv('DrRe%g_%g_validation_averaging.xtest.csv' % (Drtarget, Retarget))
    ytrain.to_csv('DrRe%g_%g_validation_averaging.ytrain.csv' % (Drtarget, Retarget))
    ytest.to_csv('DrRe%g_%g_validation_averaging.ytest.csv' % (Drtarget, Retarget))
    ypred.to_csv('DrRe%g_%g_validation_averaging.ypred.csv' % (Drtarget, Retarget))
    ypredmean.to_csv('DrRe%g_%g_validation_averaging.ypredmean.csv' % (Drtarget, Retarget))
    ypredmedian.to_csv('DrRe%g_%g_validation_averaging.ypredmedian.csv' % (Drtarget, Retarget))

