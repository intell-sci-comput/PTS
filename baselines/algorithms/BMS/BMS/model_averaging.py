import sys
import numpy as np
import pandas as pd
from optparse import OptionParser

sys.path.append('../Prior')
#from fit_prior import read_prior_par
sys.path.append('./')
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
    parser = OptionParser(usage='usage: %prog [options] DATASET')
    parser.add_option("-p", "--priorpar", dest="pparfile", default=None,
                      help="Use priors from this file (default: no priors)")
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
    parser.add_option("-k", "--kth", dest="kth", default=0,
                      type='int',
                      help="Don't do the whole cross-validation, but only the kth fold or leave-one-out point (default: 0=do all folds)")
    return parser

if __name__ == '__main__':
    # Arguments
    parser = parse_options()
    opt, args = parser.parse_args()
    dset = args[0]
    VARS = iodata.XVARS[dset]
    Y = iodata.YLABS[dset]

    # Read the data
    inFileName = '%s/data/%s' % (dset, iodata.FNAMES[dset])
    data, x, y = iodata.read_data(
        dset, ylabel=Y, xlabels=VARS, in_fname=inFileName,
    )
    if opt.pparfile != None:
        prior_par = read_prior_par(opt.pparfile)
    print x
    print y
    print prior_par
    
    # Start/end of the cross-validation
    if opt.kth < 1:
        start_end = (0, len(y))
        progressfn = '%s/model_averaging_progress__%s' % (
            dset, opt.pparfile.split('/')[-1]
        )
    else:
        start_end = (opt.kth-1, opt.kth)
        progressfn = '%s/model_averaging_progress__%04d__%s' % (
            dset, opt.kth, opt.pparfile.split('/')[-1]
        )

    # Model averaging
    npar = opt.pparfile[opt.pparfile.find('.np') + 3:]
    npar = int(npar[:npar.find('.')])
    print 'NPAR =', npar
    mse, mae = model_averaging_valid(
        x, y, VARS, prior_par, npar=npar,
        ns=opt.nsample, thin=opt.thin,
        burnin=opt.burnin,
        parallel=True,
        nT=opt.nT, sT=opt.Tf,
        par_anneal=opt.anneal,
        par_annealf=opt.annealf,
        method='lko', k=1,
        start_end=start_end,
        progressfn=progressfn,
    )
    print 'RMSE:', np.sqrt(np.mean(mse)), mse
    print 'MAE: ', np.mean(mae), mae
