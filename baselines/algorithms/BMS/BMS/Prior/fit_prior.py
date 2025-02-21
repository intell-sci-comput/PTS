import sys
from datetime import datetime
from optparse import OptionParser
from random import random, choice
sys.path.append('../')
from mcmc import *

# -----------------------------------------------------------------------------
def parse_options():
    """Parse command-line arguments.

    """
    parser = OptionParser()
    parser.add_option(
        "-s", "--source", dest="source",
        default = "named_equations",
        help="formula dataset to use ('full' or 'named_equations' (default))"
    )
    parser.add_option("-n", "--nvar", dest="nvar", type="int", default=5,
                      help="number of variables to include (default 5)")
    parser.add_option("-m", "--npar", dest="npar", type="int", default=None,
                      help="number of parameters to include (default: 2*NVAR)")
    parser.add_option("-f", "--factor", dest="fact", type="float", default=.05,
                      help="factor for the parameter adjustment (default 0.05)")
    parser.add_option("-r", "--repetitions", type="int", default=1000000,
                      dest="nrep",
                      help="formulas to generate between parameter updates")
    parser.add_option("-M", "--maxsize", type="int", default=50,
                      dest="max_size",
                      help="maximum tree (formula) size")
    parser.add_option("-c", "--continue", dest="contfile", default=None,
                      help="continue from parameter values in CONTFILE (default: start from scratch)")
    parser.add_option("-q", "--quadratic",
                      action="store_true", dest="quadratic", default=False,
                      help="fit parameters for quadratic terms (default: False)")
    return parser


# -----------------------------------------------------------------------------
def read_target_values(source, quadratic=False):
    """Read the target proportions for each type of operation.

    """
    # Number of formulas
    infn1 = '../Process-Formulas/data/%s.wiki.parsed__num_operations.dat' % (source)
    with open(infn1) as inf1:
        lines = inf1.readlines()
        nform = sum([int(line.strip().split()[1]) for line in lines])
    # Fraction of each of the operations
    infn2 = '../Process-Formulas/data/%s.wiki.parsed__operation_type.dat' % (source)
    with open(infn2) as inf2:
        lines = inf2.readlines()
        target = dict([('Nopi_%s' % line.strip().split()[0],
                        float(line.strip().split()[1]) / nform)
                       for line in lines])
    # Fraction of each of the operations squared
    if quadratic:
        infn3 = '../Process-Formulas/data/%s.wiki.parsed__operation_type_sq.dat' % (source)
        with open(infn3) as inf3:
            lines = inf3.readlines()
            target2 = dict([('Nopi2_%s' % line.strip().split()[0],
                            float(line.strip().split()[1]) / nform)
                           for line in lines])
        for k, v in list(target2.items()):
            target[k] = v
    # Done
    return target, nform


# -----------------------------------------------------------------------------
def update_ppar(tree, current, target, terms=None, step=0.05):
    """ Update the prior parameters using a gradient descend of sorts.

    """

    # Which terms should we update? (Default: all)
    if terms == None:
        terms = list(current.keys())
    # Update
    for t in terms:
        if current[t] > target[t]:
            tree.prior_par[t] += min(
                .5,
                random() * step * float(current[t] - target[t]) / \
                (target[t] + 1e-10)
            )
        elif current[t] < target[t]:
            tree.prior_par[t] -= min(
                .5,
                random() * step * float(target[t] - current[t]) / \
                (target[t] + 1e-10)
            )
        else:
            pass
    # Make sure quadratic terms are not below the minimum allowed
    for t in [t for t in terms if t.startswith('Nopi2_')]:
        """
        lint = t.replace('Nopi2_', 'Nopi_')
        op = t[6:]
        nopmax = float(tree.max_size) / tree.ops[op] - 1.
        minval = - tree.prior_par[lint] / nopmax
        """
        minval = 0.0
        if tree.prior_par[t] < minval:
            tree.prior_par[t] = minval
    
    return


# -----------------------------------------------------------------------------
def read_prior_par(inFileName):
    with open(inFileName) as inf:
        lines = inf.readlines()
    ppar = dict(list(zip(lines[0].strip().split()[1:],
                    [float(x) for x in lines[-1].strip().split()[1:]])))
    return ppar

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    MAX_SIZE = 50
    parser = parse_options()
    opt, args = parser.parse_args()
    if opt.npar == None:
        opt.npar = 2 * opt.nvar
    target, nform = read_target_values(opt.source, quadratic=opt.quadratic)
    print(opt.contfile)
    print('\n>> TARGET:', target)

    # Create prior parameter dictionary from scratch or load it from file
    if opt.contfile != None:
        ppar = read_prior_par(opt.contfile)
        # Add values to parameters for the quadratic terms (and modify
        # those of the linear terms accordingly) if you loaded ppar
        # from a file without quadratic terms
        if opt.quadratic:
            for t in [t for t in target
                      if t.startswith('Nopi2_') and t not in list(ppar.keys())]:
                ppar[t] = 0.
                """
                lint = t.replace('Nopi2_', 'Nopi_')
                op = t[6:]
                nopmax = float(MAX_SIZE) / OPS[op] - 1.
                ppar[lint] /= (1. - float(target[lint]) / nopmax)
                minval = - ppar[lint] / nopmax 
                ppar[t] = .5 * minval
                """ 

    else:
        ppar = dict([(k, 10.0) for k in target if k.startswith('Nopi_')] +
                    [(k, 0.0) for k in target if not k.startswith('Nopi_')])
    print('\n>> PRIOR_PAR:', ppar)

    # Preliminaries
    if opt.quadratic:
        outFileName = 'prior_param_sq.%s.nv%d.np%d.maxs%d.%s.dat' % (
            opt.source, opt.nvar, opt.npar, opt.max_size, datetime.now(),
        )
    else:
        outFileName = 'prior_param.%s.nv%d.np%d.maxs%d.%s.dat' % (
            opt.source, opt.nvar, opt.npar, opt.max_size, datetime.now(),
        )
    with open(outFileName, 'w') as outf:
        print('#', ' '.join([o for o in ppar]), file=outf)
    iteration = 0

    # Do the loop!
    while True:
        # Create new seed formula
        tree = Tree(
            ops = dict([(o[5:], OPS[o[5:]])
                        for o in ppar if o.startswith('Nopi_')]),
            variables=['x%d' % (i+1) for i in range(opt.nvar)],
            parameters=['a%d' % (i+1) for i in range(opt.npar)],
            max_size=opt.max_size,
            prior_par=ppar,
        )

        # Generate the formulas and compute the features
        current = dict([(t, 0) for t in ppar])
        for rep in range(opt.nrep):
            tree.mcmc_step()
            for o, nopi in list(tree.nops.items()):
                current['Nopi_%s' % o] += nopi
                try:
                    current['Nopi2_%s' % o] += nopi * nopi
                except KeyError:
                    pass

        # Normalize the current counts
        current = dict([(t, float(v) / opt.nrep) for t, v in list(current.items())])
        
        # Output some info to stdout and to output file
        print(40 * '-')
        print(tree.prior_par)
        with open(outFileName, 'a') as outf:
            print(iteration, ' '.join([str(v) for v in list(ppar.values())]), file=outf)
        for t in ppar:
            print(t, current[t], target[t], \
                '%.1f' % (float(current[t] - target[t]) * 100. / target[t]))
        iteration += 1

        # Update parameters
        dice = random()
        if dice < .8: # all terms
            update_ppar(tree, current, target, step=opt.fact)
        else:         # a single randomly chosen term
            update_ppar(tree, current, target, step=opt.fact,
                        terms=[choice(list(current.keys()))])
        ppar = tree.prior_par
