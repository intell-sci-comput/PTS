from .base import TokenGenerator
import yaml
import numpy as np
import random
import math
import sympy


class Random_TokenGenerator(TokenGenerator):
    def __init__(
        self,
        regressor,
        config,
        variables,
        operators_op,
        use_const,
        n_inputs,
        use_extra_const=False,
    ):

        self.config = config
        self.variables = variables
        self.operators_op = operators_op
        self.use_const = use_const
        self.n_inputs = n_inputs

        self.max_depths = config["maxdepth"]
        self.n_const = config["n_const"]
        self.const_range = config["const_range"]
        self.depths = list(range(1, self.max_depths + 1))

    def step(
        self,
        n_psrn_tokens,
        n_sample_variables,
        X,
        y,
        use_set=True,
        reset=False,
        use_float_const=False,
    ):
        tokens = []
        dec = self.n_const if use_float_const else 0
        for i in range(n_psrn_tokens - n_sample_variables - dec):
            token = self.get_sampled_tree(
                self.depths, self.operators_op, self.variables
            )
            tokens.append(token)

        print("n_sample_variables", n_sample_variables)
        print("n_psrn_tokens", n_psrn_tokens)
        print("tokens:", tokens)

        for i in range(dec):
            tokens.append(
                round(random.uniform(self.const_range[0], self.const_range[1]), 1)
            )

        return "nan", tokens

    def reward(self, reward, expressions):
        return

    def get_sampled_tree(self, depth_ls, operator_op_ls, root_expr_ls):

        def create_expr_tree(current_depth):
            if current_depth == depth:
                idx = np.random.choice(len(root_expr_ls))
                return root_expr_ls[idx]
            op = np.random.choice(operator_op_ls)
            if op.is_unary:
                expr = create_expr_tree(current_depth + 1)
                new_expr = op.get_expr(expr)
            else:
                expr1 = create_expr_tree(current_depth + 1)
                expr2 = create_expr_tree(current_depth + 1)
                new_expr = op.get_expr(expr1, expr2)
            return new_expr

        depth = np.random.choice(depth_ls)
        final_expr = create_expr_tree(0)
        return final_expr
