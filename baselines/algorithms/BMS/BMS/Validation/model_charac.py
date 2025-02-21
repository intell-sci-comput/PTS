import sys
import ast
import numpy as np
from scipy.stats import sem
from sklearn import cross_validation
from sympy import sympify, latex
from optparse import OptionParser

sys.path.append('..')
from mcmc import Tree
from iodata import *

sys.path.append('../Prior')
from fit_prior import read_prior_par


# -----------------------------------------------------------------------------
def parse_options():
    """Parse command-line arguments.

    """
    parser = OptionParser(usage='usage: %prog [options] DATASET MODELSTR')
    parser.add_option("-p", "--priorpar", dest="pparfile", default=None,
                      help="Use priors from this file (default: no priors)")
    parser.add_option("-v", "--parvalues", dest="par_values", default=None,
                      help="Use these parameter values (in dictionary format)")
    parser.add_option("-l", "--loo",
                      action="store_false", dest="loo", default=True,
                      help="don't do the leave one out cross-validation")
    return parser

# -----------------------------------------------------------------------------
def cross_val(t, method='kfold', k=2):
    x, y = t.x, t.y
    if method == 'lko':
        ttsplit = cross_validation.LeavePOut(len(y), k)
    elif method == 'kfold':
        ttsplit = cross_validation.KFold(len(y), n_folds=k)
    else:
        raise ValueError
    serr, aerr = [], []
    for train_index, test_index in ttsplit:
        xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
        tt = Tree(x=xtrain, y=ytrain, from_string=str(t))
        ypred = tt.predict(xtest)
        serr.append(np.mean((ytest - ypred)**2))
        aerr.append(np.mean(np.abs(ytest - ypred)))
    return serr, aerr


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = parse_options()
    opt, args = parser.parse_args()
    if opt.par_values != None:
        opt.par_values = ast.literal_eval(opt.par_values)

    dset = args[0]
    modelstr = args[1]

    data, x, y = read_data(dset)

    t = Tree(from_string=modelstr)
    t.x, t.y = x, y
    if opt.pparfile:
        t.prior_par = read_prior_par(opt.pparfile)
    if opt.par_values != None:
        t.par_values = opt.par_values
        sse = t.get_sse(fit=False)
    else:
        sse = t.get_sse(fit=True)
    bic = t.get_bic(reset=True, fit=False)
    if opt.loo:
        serr, aerr = cross_val(t, method='lko', k=1)

    # Output
    print '-' * 80
    print 'Dataset: ', dset
    print 'Model:   ', t
    print 'LaTeX:   ', latex(sympify(t.canonical()))
    print 'Param:   ', t.par_values

    print 'MAE:     ', np.sum(np.abs(y - t.predict(x))) / len(y)
    print 'SSE:     ', sse
    print 'RMSE:    ', np.sqrt(sse / float(len(y)))
    print 'BIC:     ', bic

    if opt.pparfile:
        print 'Prior_c: ', t.prior_par
    print '-log(F): ', t.get_energy(bic=False, reset=False)

    if opt.loo:
        print 'LOO-RMSE:', np.sqrt(np.mean(serr))
        print 'LOO-MSE:  %g+-%g' % (np.mean(serr), sem(serr))
        print 'LOO-MAE:  %g+-%g' % (np.mean(aerr), sem(aerr))
    print '-' * 80

        
