import sys
import pandas as pd
from copy import deepcopy
from optparse import OptionParser

sys.path.append('../')
from mcmc import *
from parallel import *

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
    parser.add_option("-n", "--nsample", dest="nsample", default=100000,
                      type='int',
                      help="Number of samples (default: 100000)")
    parser.add_option("-t", "--thin", dest="thin", default=10,
                      type='int',
                      help="Thinning of the sample (default: 10)")
    parser.add_option("-b", "--burnin", dest="burnin", default=1000,
                      type='int',
                      help="Burn-in (default: 1000)")
    parser.add_option("-T", "--nT", dest="nT", default=10,
                      type='int',
                      help="Number of temperatures (default: 10)")
    parser.add_option("-s", "--Tf", dest="Tf", default=1.2,
                      type='float',
                      help="Factor between temperatures (default: 1.20)")
    parser.add_option("-a", "--anneal", dest="anneal", default=100,
                      type='int',
                      help="Annealing threshold. If there are no tree swaps for more than this number of steps, the parallel tempering is annealed (default: 100)")
    parser.add_option("-f", "--annealf", dest="annealf", default=5,
                      type='float',
                      help="Annealing factor: all temperatures are multiplied by this factor during the heating phase of the annealing (default: 5)")
    return parser

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Arguments
    parser = parse_options()
    opt, args = parser.parse_args()
    dset = args[0]

    VARS = iodata.XVARS[dset]
    Y = iodata.YLABS[dset]

    # Temperatures
    Ts = [1] + [opt.Tf**k for k in range(1, opt.nT)]

    # Read the data
    inFileName = '%s/data/%s' % (dset, iodata.FNAMES[dset])
    data, x, y = iodata.read_data(
        dset, ylabel=Y, xlabels=VARS, in_fname=inFileName,
    )
    if opt.pparfile != None:
        prior_par = read_prior_par(opt.pparfile)

    # Get the output file ready
    try:
        outfname = '%s/sample_charac_parallel__%s' % (
            dset, opt.pparfile.split('/')[-1]
        )
        npar = outfname[outfname.find('.np') + 3:]
        npar = int(npar[:npar.find('.')])
    except AttributeError:
        outfname = '%s/sample_charac_parallel' % (dset)
    outf = open(outfname, 'w')
    print >> outf, '#', prior_par
    
    # Initialize
    p = Parallel(
        Ts,
        variables=VARS,
        parameters=['a%d' % i for i in range(npar)],
        x=x, y=y,
        prior_par=prior_par,
    )
    print >> outf, '# Ts:        ', Ts
    print >> outf, '# Variables: ', p.t1.variables
    print >> outf, '# Options:   ', opt
    print >> outf, '# Parameters:', p.t1.parameters, '\n#'

    # Burnin
    for i in range(opt.burnin):
        p.mcmc_step()
        BT1, BT2 = p.tree_swap()

    # MCMC
    vprob = dict([(v, 0) for v in VARS])
    sizeprob = dict([(s, 0) for s in range(p.t1.max_size+1)])
    nparprob = dict([(n, 0) for n in range(npar+1)])
    last_swap = dict([(T, 0) for T in p.Ts[:-1]])
    max_inactive_swap = 0
    for i in range(opt.nsample):
        # Output
        print >> sys.stdout, i+1
        print >> outf, ' || '.join(
            [str(tmp) for tmp in [i+1, p.t1.E, p.t1.get_energy()[0],
                                  p.t1.EB*2, p.t1.bic,
                                  p.t1.size, p.t1.canonical(),
                                  p.t1, p.t1.par_values, max_inactive_swap]]
        )
        outf.flush()
        # MCMC updates
        for kk in range(opt.thin):
            p.mcmc_step()
            BT1, BT2 = p.tree_swap()
            if BT1 != None:
                last_swap[BT1] = i
        # Keep track of some stuff
        for v in VARS:
            if v in [n.value for n in p.t1.ets[0]]:
                vprob[v] += 1
        nparprob[len(set([n.value for n in p.t1.ets[0]
                          if n.value in p.t1.parameters]))] += 1
        sizeprob[p.t1.size] += 1
        # Anneal if the some configuration is stuck
        max_inactive_swap = max([i-last_swap[T] for T in last_swap])
        if max_inactive_swap > opt.anneal:
            p.anneal(n=opt.anneal*opt.thin, factor=opt.annealf)
            last_swap = dict([(T, i) for T in p.Ts[:-1]])
        
    # Report
    print >> outf, '\n# formula size' 
    for s in sizeprob:
        print >> outf, s, sizeprob[s] / float(opt.nsample)

    print >> outf, '\n# variable use' 
    for v in vprob:
        print >> outf, v, vprob[v] / float(opt.nsample)

    print >> outf, '\n# parameters' 
    for n in nparprob:
        print >> outf, n, nparprob[n] / float(opt.nsample)

    # Done
    outf.close()
