import sys
import pandas as pd
from copy import deepcopy
from optparse import OptionParser

sys.path.append('../')
from mcmc import *

sys.path.append('../Prior')
from fit_prior import read_prior_par

import iodata

# -----------------------------------------------------------------------------
def parse_options():
    """Parse command-line arguments.

    """
    parser = OptionParser(usage='usage: %prog [options] DATASET')
    parser.add_option("-p", "--priorpar", dest="pparfile", default=None,
                      help="Use priors from this file (default: no priors)")
    return parser

# -----------------------------------------------------------------------------
NS = 1000000
THIN = 1

# -----------------------------------------------------------------------------
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

    # Get the output file ready
    try:
        outfname = '%s/sample_charac__%s' % (dset, opt.pparfile.split('/')[-1])
        npar = outfname[outfname.find('.np') + 3:]
        npar = int(npar[:npar.find('.')])
    except AttributeError:
        outfname = '%s/sample_charac' % (dset)
    outf = open(outfname, 'w')
    print >> outf, '#', prior_par
    
    # Initialize
    t = Tree(
        variables=VARS,
        parameters=['a%d' % i for i in range(npar)],
        x=x, y=y,
        prior_par=prior_par,
    )
    ##print >> outf, '# Fit parameters:', t.fit_par
    print >> outf, '# Variables: ', t.variables
    print >> outf, '# Parameters:', t.parameters, '\n#'

    # Burnin
    #for i in range(100):
    #    t.mcmc_step()

    # MCMC
    vprob = dict([(v, 0) for v in VARS])
    sizeprob = dict([(s, 0) for s in range(t.max_size+1)])
    nparprob = dict([(n, 0) for n in range(npar+1)])
    for i in range(NS):
        print >> sys.stdout, i+1
        print >> outf, ' || '.join(
            [str(tmp) for tmp in [i+1, t.E, t.get_energy()[0], t.EB*2, t.bic, t.size, t.canonical(), t, t.par_values]]
        )
        outf.flush()
        """
        t.BT = 10.
        for kk in range(THIN/3):
            t.mcmc_step()
        t.BT = 1.
        t.get_energy(bic=True, reset=True)
        """
        for kk in range(THIN):
            t.mcmc_step()
        for v in VARS:
            if v in [n.value for n in t.ets[0]]:
                vprob[v] += 1
        nparprob[len(set([n.value for n in t.ets[0]
                          if n.value in t.parameters]))] += 1
        sizeprob[t.size] += 1

    """
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    # 2nd MCMC to CHECK IF TRAPPING IS DUE TO REPRESENTATIVE REDEFINITION
    t2 = Tree(
        variables=VARS,
        parameters=['a%d' % i for i in range(npar)],
        x=x, y=y,
        prior_par=prior_par,
    )
    t2.representative = deepcopy(t.representative)
    vprob = dict([(v, 0) for v in VARS])
    sizeprob = dict([(s, 0) for s in range(t2.max_size+1)])
    nparprob = dict([(n, 0) for n in range(npar+1)])
    for i in range(3*NS):
        print >> sys.stdout, i+1
        print >> outf, ' || '.join(
            [str(tmp) for tmp in [i+1, t2.E, t2.get_energy(), t2.bic, t2.canonical(), t2, t2.par_values]]
        )
        outf.flush()
        for kk in range(THIN):
            t2.mcmc_step()
        for v in VARS:
            if v in [n.value for n in t2.ets[0]]:
                vprob[v] += 1
        nparprob[len(set([n.value for n in t2.ets[0]
                          if n.value in t2.parameters]))] += 1
        sizeprob[t2.size] += 1
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    """
        
    # Report
    print >> outf, '\n# formula size' 
    for s in sizeprob:
        print >> outf, s, sizeprob[s] / float(NS)

    print >> outf, '\n# variable use' 
    for v in vprob:
        print >> outf, v, vprob[v] / float(NS)

    print >> outf, '\n# parameters' 
    for n in nparprob:
        print >> outf, n, nparprob[n] / float(NS)

    # Done
    outf.close()
