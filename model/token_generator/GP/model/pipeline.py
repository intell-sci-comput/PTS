from collections import Counter
from time import time

from .expr_utils.utils import FinishException, time_limit
from .ga.ga import GAPipeline
from .config import Config
from .expr_utils.epression_to_tokens import expression_to_tokens

import sympy as sp

class Pipeline:
    def __init__(self, config: Config = None):
        self.config: Config = config
        if config is None:
            self.config = Config()
            self.config.init()
        self.expr_form = ""
        self.ga1, self.ga2 = None, None
        self.msdb = None

        self.tms = 0

        self.from_psrn = []

    def fit(self, x=None, t=None, x_test=None, t_test=None, clear=True):
        if self.config.x is None:
            self.config.set_input(x=x, t=t, x_=x_test, t_=t_test)

        if clear:
            self.ga1, self.ga2 = GAPipeline(self.config), GAPipeline(self.config)
            self.sym_tol1 = []
            self.sym_tol2 = []
        else:
            pass

        # self.all_forms = []

        try:
            tm_start = time()
            self.tms += 1

            if self.config.verbose:
                print(
                    f'\rEpisode: {self.tms + 1}/{self.config.epoch}, time: {round(time() - tm_start, 2)}s, '
                    f'expression: {self.config.best_exp[0]}, loss: {self.config.best_exp[1]}, '
                    f'form: {self.expr_form}F', end='')
            
            # self.all_forms.append(self.expr_form)
            
            pop = None

            if self.tms % 25 == 0:
                self.sym_tol1.clear()
            if self.tms % 30 == 0:
                self.sym_tol2.clear()
            if self.tms % 10 <= 8:
                pop = self.ga1.ga_play(self.sym_tol1)
                self.sym_tol1 += pop
                pop = self.ga2.ga_play(self.sym_tol2)
                self.sym_tol2 += pop
            if self.tms % 10 >= 7:
                self.sym_tol2 += self.ga2.ga_play(self.sym_tol2)
            if self.tms % 10 == 5:
                self.ga1.ga_play(self.sym_tol1)
            print('self.config.pareto',self.config.pareto)

            symbols = []
            symbols.append(self.config.best_exp[0])
            # symbols.append(self.expr_form)
            pareto_exprs = [tup[1] for tup in self.config.pareto]
            symbols += pareto_exprs
            symbols += self.from_psrn

            print('symbols')
            print(symbols)

            return self.config.best_exp[0], symbols

        except FinishException:
            pass
        return self.config.best_exp[0], []


    def use_psrn_reward_expressions_to_update(self, expressions):

        self.from_psrn = expressions

        pop = []

        for expr in expressions:
            try:
                tok = expression_to_tokens(str(sp.S(expr).evalf(3)), self.config) 
                pop.append(tok)
            except Exception:
                continue

        pop = self.ga1.ga_play(pop)

        self.sym_tol1 += pop

        