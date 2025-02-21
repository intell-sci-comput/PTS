import sys
import pandas as pd
from optparse import OptionParser

sys.path.append('../')
from mcmc import *

sys.path.append('../Prior')
from fit_prior import read_prior_par

from data_analysis import post_SA

import iodata

# -----------------------------------------------------------------------------
def parse_options():
    """Parse command-line arguments.

    """
    parser = OptionParser(usage='usage: %prog [options] DATASET')
    parser.add_option("-p", "--priorpar", dest="pparfile", default=None,
                      help="Use priors from this file (default: no priors)")
    parser.add_option("-n", "--nstep", dest="nstep", default=1000,
                      type='int',
                      help="Number of steps at each temperature (default: 1000)")
    parser.add_option("-f", "--annealf", dest="annealf", default=.95,
                      type='float',
                      help="Annealing factor (default: 0.95)")
    return parser

# -----------------------------------------------------------------------------
if __name__ == '__main__':
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
        npar = opt.pparfile[opt.pparfile.find('.np') + 3:]
        npar = int(npar[:npar.find('.')])
    print x
    print y
    print prior_par
    print 'Npar =', npar

    # SA
    t = post_SA(x, y, VARS, prior_par, npar=npar, ns=opt.nstep,
                T_ini=2., T_fin=0.01, T_sched=opt.annealf)
    print 'Most plausible F:', t
