from json import load

import numpy as np

from .expr_utils.utils import expression_dict


class Config:
    def __init__(self):
        self.symbol_tol_num = 0
        self.best_exp = None, 1e999

        self.x = None
        self.x_ = None
        self.t = None
        self.t_ = None
        self.has_const = None
        self.const_optimize = None
        self.exp_dict = None
        self.reward_end_threshold = None
        self.verbose = None
        self.num_of_var = None
        self.epoch = None
        self.tokens = None
        self.pareto = None

        class gp:
            def __init__(self):
                self.tournsize = None
                self.max_height = None
                self.cxpb = None
                self.mutpb = None
                self.max_const = None
                self.pops = None
                self.times = None
                self.hof_size = None
                self.token_discount = None

        self.gp = gp()

    def set_input(self, *, x, t, x_, t_, tokens):
        self.x = np.array(x)
        self.x_ = np.array(x_)
        self.t = np.array(t)
        self.t_ = np.array(t_)
        self.num_of_var = self.x.shape[0]
        self.tokens = tokens
        self.exp_dict = expression_dict(self.tokens, self.num_of_var, self.has_const)
        self.pareto = []

    def config_base(self, *, epoch=100, has_const=True, const_optimize=True, tokens=None, verbose=False,
                    reward_end_threshold=1e-10):


        print('config_base got tokens:', tokens)

        self.epoch = epoch
        self.const_optimize = const_optimize
        self.has_const = has_const
        self.tokens = tokens
        self.verbose = verbose
        self.reward_end_threshold = reward_end_threshold

    def config_gp(self, *, max_const=5, pops=500, times=30, tournsize=10, max_height=10, cxpb=0.1, mutpb=0.5,
                  hof_size=20, token_discount=0.99):
        self.gp.max_height = max_height
        self.gp.tournsize = tournsize
        self.gp.cxpb = cxpb
        self.gp.mutpb = mutpb
        self.gp.max_const = max_const
        self.gp.pops = pops
        self.gp.times = times
        self.gp.hof_size = hof_size
        self.gp.token_discount = token_discount

    def init(self):
        self.config_gp()
        self.config_base()

    # def json(self, filepath):
    #     with open(filepath, 'r') as f:
    #         js = load(f)
    #         self.config_base(**js['base'])
    #         self.config_gp(**js['gp'])
    
    def json(self, input_data):
        if isinstance(input_data, str):
            # If it's a string, assume it's a file path
            with open(input_data, 'r') as f:
                js = load(f)
        elif isinstance(input_data, dict):
            # If it's a dict, use it directly
            js = input_data
        else:
            raise ValueError("Invalid input: expected file path or dict")

        # Assuming the input always has 'gp' key and no 'base' key
        self.config_gp(**js['gp'])
            
    def from_dict(self, dict):
        js = dict
        self.config_base(**js['base'])
        self.config_gp(**js['gp'])
