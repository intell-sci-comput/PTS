import sys
import time
import sympy
import numpy as np
from copy import deepcopy
from random import seed, random, randint
from numpy import exp
from .mcmc import *

from .utils.sympy_utils import set_real, del_float_one, prun_constant, my_equals

def from_bms_model_get_final_expr(tree):
    final_model = deepcopy(tree)
    expr_best = str(final_model)
    expr_best_replaced = expr_best
    for k,v in final_model.par_values['d0'].items():
        expr_best_replaced = expr_best_replaced.replace(k,'('+str(v)+')')
    return expr_best_replaced # str

def check_bms_sympy_probe(bms_tree_p, probe_evalf, is_positive):
    mbs_expr_str = from_bms_model_get_final_expr(bms_tree_p)
    mbs_expr_str = sympy.sympify(mbs_expr_str).expand()
    mbs_expr_str = prun_constant(mbs_expr_str, n_digits=2)
    return my_equals(mbs_expr_str, probe_evalf, is_positive)

class Parallel():
    """ The Parallel class for parallel tempering. """

    # -------------------------------------------------------------------------
    def __init__(self, Ts, ops=OPS, variables=['x'], parameters=['a'],
                 max_size=50,
                 prior_par={}, x=None, y=None, bms_probe=None, bms_ispositive=None):
        # All trees are initialized to the same tree but with different BT
        Ts.sort()
        self.Ts = [str(T) for T in Ts]
        self.trees = {'1' : Tree(ops=ops,
                                 variables=deepcopy(variables),
                                 parameters=deepcopy(parameters),
                                 prior_par=deepcopy(prior_par), x=x, y=y,
                                 max_size=max_size,
                                 BT=1)}
        self.t1 = self.trees['1']
        for BT in [T for T in self.Ts if T != 1]:
            treetmp = Tree(ops=ops,
                           variables=deepcopy(variables),
                           parameters=deepcopy(parameters),
                           prior_par=deepcopy(prior_par), x=x, y=y,
                           root_value=str(self.t1),
                           max_size=max_size,
                           BT=float(BT))
            self.trees[BT] = treetmp
            # Share fitted parameters and representative with other trees
            self.trees[BT].fit_par = self.t1.fit_par
            self.trees[BT].representative = self.t1.representative

        self.bms_probe = bms_probe
        self.bms_ispositive = bms_ispositive
        return

    # -------------------------------------------------------------------------
    def mcmc_step(self, verbose=False, p_rr=0.05, p_long=.45):
        """ Perform a MCMC step in each of the trees. """
        # Loop over all trees
        for T, tree in list(self.trees.items()):
            # MCMC step
            tree.mcmc_step(verbose=verbose, p_rr=p_rr, p_long=p_long)
        self.t1 = self.trees['1']

        if self.bms_ispositive is not None and self.bms_probe is not None:
            if check_bms_sympy_probe(self.t1, probe_evalf=self.bms_probe, is_positive=self.bms_ispositive):
                print('Found !!!!! sympy probe')
                print(self.bms_probe)
                # print('=')
                return True
        # Done
        return False

    # -------------------------------------------------------------------------
    def tree_swap(self):
        # Choose Ts to swap
        nT1 = randint(0, len(self.Ts)-2)
        nT2 = nT1 + 1
        t1 = self.trees[self.Ts[nT1]]
        t2 = self.trees[self.Ts[nT2]]
        # The temperatures and energies
        BT1, BT2 = t1.BT, t2.BT
        EB1, EB2, EP1, EP2 = t1.EB, t2.EB, t1.EP, t2.EP
        # The energy change
        DeltaE = np.float(EB1) * (1./BT2 - 1./BT1) + \
                 np.float(EB2) * (1./BT1 - 1./BT2)
        if DeltaE > 0:
            paccept = exp(-DeltaE)
        else:
            paccept = 1.
        # Accept/reject change
        if random() < paccept:
            self.trees[self.Ts[nT1]] = t2
            self.trees[self.Ts[nT2]] = t1
            t1.BT = BT2
            t2.BT = BT1
            self.t1 = self.trees['1']
            return self.Ts[nT1], self.Ts[nT2]
        else:
            return None, None

    # -------------------------------------------------------------------------
    def anneal(self, n=1000, factor=5, nowtime=None, time_budget=None):
        
        import time
        start_time = time.time()
        
        # Heat up
        for t in list(self.trees.values()):
            t.BT *= factor
        for kk in range(n):
            
            if nowtime is not None and time_budget is not None:
                if time.time() - start_time + nowtime > time_budget:
                    return
            
            print('# Annealing heating at %g: %d / %d' % (
                self.trees['1'].BT, kk, n
            ), file=sys.stderr)
            r = self.mcmc_step()
            if r:
                return
            self.tree_swap()
        # Cool down (return to original temperatures)
        for BT, t in list(self.trees.items()):
            t.BT = float(BT)
        for kk in range(2*n):
            
            if nowtime is not None and time_budget is not None:
                if time.time() - start_time + nowtime > time_budget:
                    return
            
            print('# Annealing cooling at %g: %d / %d' % (
                self.trees['1'].BT, kk, 2*n
            ), file=sys.stderr)
            r = self.mcmc_step()
            if r:
                return
            self.tree_swap()
        # Done
        return

    # -------------------------------------------------------------------------
    def trace_predict(self, x,
                      burnin=5000, thin=100, samples=10000,
                      anneal=100, annealf=5, verbose=True,
                      write_files=True,
                      progressfn='progress.dat', reset_files=True,
                      middle_time=None,
                      time_budget=None):
        
        start_time = time.time()
        
        # Burnin
        if verbose:
            sys.stdout.write('# Burning in\t')
            sys.stdout.write('[%s]' % (' ' * 50))
            sys.stdout.flush()
            sys.stdout.write('\b' * (50+1))
        for i in range(burnin):
            r = self.mcmc_step()
            if r:
                ypred[s] = self.trees['1'].predict(x)
                return pd.DataFrame.from_dict(ypred)
            if verbose and (i % (burnin / 50) == 0):
                sys.stdout.write('=')
                sys.stdout.flush()
        # MCMC
        if write_files:
            if reset_files:
                progressf = open(progressfn, 'w')
            else:
                progressf = open(progressfn, 'a')
        if verbose:
            sys.stdout.write('\n# Sampling\t')
            sys.stdout.write('[%s]' % (' ' * 50))
            sys.stdout.flush()
            sys.stdout.write('\b' * (50+1))
        ypred = {}
        last_swap = dict([(T, 0) for T in self.Ts[:-1]])
        max_inactive_swap = 0
        for s in range(samples):
            # MCMC updates
            ready = False
            while not ready:
                for kk in range(thin):
                    r = self.mcmc_step()
                    if r:
                        ypred[s] = self.trees['1'].predict(x)
                        return pd.DataFrame.from_dict(ypred)
                    BT1, BT2 = self.tree_swap()
                    if BT1 != None:
                        last_swap[BT1] = s
                # Predict for this sample (prediction must be finite;
                # otherwise, repeat
                ypred[s] = self.trees['1'].predict(x)
                ready = True not in np.isnan(np.array(ypred[s])) and \
                        True not in np.isinf(np.array(ypred[s]))
            # Output
            if verbose and (s % (samples / 50) == 0):
                sys.stdout.write('=')
                sys.stdout.flush()
            if write_files:
                # progressf.write('%s %d %s %lf %lf %d %s %s\n' % (
                #     list(x.index), s, str(list(ypred[s])),
                #     self.trees['1'].E, self.trees['1'].bic,
                #     max_inactive_swap,
                #     self.trees['1'],
                #     self.trees['1'].par_values['d0']
                # ))
                # progressf.flush()
                progressf.write('%s %s\n' % (
                    self.trees['1'],
                    self.trees['1'].par_values['d0']
                ))
                progressf.flush()
                
            
            if middle_time is not None and time_budget is not None:
                if time.time() - start_time > time_budget - middle_time:
                    # Done
                    if verbose:
                        sys.stdout.write('\n')
                        sys.stdout.flush()
                    return pd.DataFrame.from_dict(ypred)
                print(time.time() - start_time,'/',time_budget - middle_time,'(s)')
            
                
            # Anneal if the some configuration is stuck
            max_inactive_swap = max([s-last_swap[T] for T in last_swap])
            if max_inactive_swap > anneal:
                self.anneal(n=anneal*thin, factor=annealf, nowtime=time.time() - start_time, time_budget=time_budget - middle_time)
                last_swap = dict([(T, s) for T in self.Ts[:-1]])

        # Done
        if verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()
        return pd.DataFrame.from_dict(ypred)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Test main
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    sys.path.append('Validation/')
    import iodata
    sys.path.append('Prior')
    from fit_prior import read_prior_par
    from pprint import pprint

    # Temperatures
    Ts = [
        1,
        1.20,
        1.44,
        1.73,
        2.07,
        2.49,
        2.99,
        3.58,
        4.30,
        5.16,
    ]

    # Read the data
    prior_par = read_prior_par('Prior/prior_param_sq.named_equations.nv7.np7.2016-06-06 16:43:26.287530.dat')
    VARS = iodata.XVARS['Trepat']
    Y = iodata.YLABS['Trepat']
    inFileName = 'Validation/Trepat/data/%s' % (iodata.FNAMES['Trepat'])
    data, x, y = iodata.read_data(
        'Trepat', ylabel=Y, xlabels=VARS, in_fname=inFileName,
    )
    #print x, y

    # Initialize the parallel object
    p = Parallel(
        Ts,
        variables=VARS,
        parameters=['a%d' % i for i in range(7)],
        x=x, y=y,
        prior_par=prior_par,
    )

    NREP = 1000000
    for rep in range(NREP):
        print('=' * 77)
        print(rep, '/', NREP)
        p.mcmc_step()
        print('>> Swaping:', p.tree_swap())
        pprint(p.trees)
        print('.' * 77)
        for T in Ts:
            energy_ref = p.trees[T].get_energy(reset=False)[0]
            print(T, '\t',  \
                p.trees[T].E, energy_ref, \
                p.trees[T].bic)
            if abs(p.trees[T].E - energy_ref) > 1.e-6:
                print(p.trees[T].canonical(), p.trees[T].representative[p.trees[T].canonical()])
                raise
            if p.trees[T].representative != p.trees['1'].representative:
                pprint(p.trees[T].representative)
                pprint(p.trees['1'].representative)
                raise
            if p.trees[T].fit_par != p.trees['1'].fit_par:
                raise

