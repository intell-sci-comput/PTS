import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from optparse import OptionParser
from pandas import DataFrame
from random import random, choice

sys.path.append('../')
from mcmc import *

from fit_prior import read_target_values, read_prior_par

# -----------------------------------------------------------------------------
def parse_options():
    """Parse command-line arguments.

    """
    parser = OptionParser(usage='usage: %prog [options] PRIOR_PAR_FILENAME')
    parser.add_option(
        "-s", "--source", dest="source",
        default = "named_equations",
        help="formula dataset to use ('full' or 'named_equations' (default))"
    )
    parser.add_option(
        "-n", "--nvar", dest="nvar", type="int", default=-1,
        help="number of varaibles (default specied by PRIOR_PAR_FILENAME)",
    )
    parser.add_option(
        "-t", "--thin", dest="thin", type="int", default=20,
        help="thining of the sample (default: 20)",
    )
    parser.add_option(
        "-m", "--npar", dest="npar", type="int", default=-1,
        help="number of parameters (default specied by PRIOR_PAR_FILENAME)",
    )
    parser.add_option("-r", "--nrep", dest="nrep", type="int", default=1000,
                      help="number of repetitions (default 1000)")
    parser.add_option("-q", "--quadratic",
                      action="store_true", dest="quadratic", default=False,
                      help="compare also quadratic terms (default: False)")
    return parser


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Parse command-line arguments
    parser = parse_options()
    opt, args = parser.parse_args()
    pparFileName = args[0]
    if opt.nvar < 1:
        tmp = pparFileName[pparFileName.find('.nv') + 3:]
        tmp = tmp[:tmp.find('.')]
        opt.nvar = int(tmp)
    if opt.npar < 1:
        tmp = pparFileName[pparFileName.find('.np') + 3:]
        tmp = tmp[:tmp.find('.')]
        opt.npar = int(tmp)
    ppar = read_prior_par(pparFileName)
    if opt.quadratic:
        target, nform = read_target_values(opt.source, quadratic=True)
    else:
        target, nform = read_target_values(opt.source, quadratic=False)

    print '> ppar =', ppar
    print '> n =', opt.nvar, '; m =', opt.npar
    print '> source =', opt.source
    print '> nform =', nform

    # Initialize the counter of numbers of operations
    if opt.quadratic:
        term_count = dict(
            [(o, [0] * opt.nrep) for o in ppar if o.startswith('Nopi_')] +
            [('Nopi2_%s' % o[5:], [0] * opt.nrep) for o in ppar
             if o.startswith('Nopi_')]
        )
    else:
        term_count = dict(
            [('Nopi_%s' % o, [0] * opt.nrep) for o in OPS]
        )

    # Do the repetitions of the sampling
    for rep in range(opt.nrep):
        # Create new formula
        tree = Tree(
            ops = dict([(o[5:], OPS[o[5:]])
                        for o in ppar if o.startswith('Nopi_')]),
            variables=['x%d' % (i+1) for i in range(opt.nvar)],
            parameters=['a%d' % (i+1) for i in range(opt.npar)],
            prior_par=ppar,
        )

        # Sample the formulas and compute the features
        for n in range(nform):
            for t in range(opt.thin): # thinning
                tree.mcmc_step()
            for o, nopi in tree.nops.items():
                term_count['Nopi_%s' % o][rep] += nopi
                if opt.quadratic:
                    term_count['Nopi2_%s' % o][rep] += nopi*nopi

        # Some info to stderr
        print >> sys.stderr, rep+1, tree.size, tree
                
    # Plot the distributions
    sns.set_style("whitegrid")

    df = DataFrame.from_dict(term_count)
    print df
    fig = df.hist(color='k', alpha=0.5, bins=30, figsize=(30, 20))
    for row in fig:
        for panel in row:
            try:
                t = panel.get_title()
                panel.plot((target[t] * nform, target[t] * nform),
                           panel.get_ylim(), linewidth=3, color='red')
                panel.plot((np.mean(term_count[t]), np.mean(term_count[t])),
                           panel.get_ylim(), color='blue')
                if t in ppar:
                    panel.set_title('%s (%lf)' % (t, ppar[t]))
                else:
                    panel.set_title('%s (0)' % (t))
            except KeyError:
                pass

    plotFileName = '%s___%s.nv%d.np%d.pdf' % (
        pparFileName,
        opt.source,
        opt.nvar,
        opt.npar,
    )
    plt.savefig(plotFileName)
    plt.close()

    csvFileName = '%s___%s.nv%d.np%d.csv' % (
        pparFileName,
        opt.source,
        opt.nvar,
        opt.npar,
    )
    df.to_csv(csvFileName)
