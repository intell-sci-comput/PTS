from .base import TokenGenerator
import yaml
import numpy as np
import random
import math
import sympy

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
triu_ls = []
for i in range(100):
    triu_ls.append(
        torch.triu_indices(i + 1, i + 1, offset=0, dtype=torch.long, device=device)
    )

N = 1
last_node = None

class MCTS_TokenGenerator(TokenGenerator):
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

        self.max_depth = n_inputs

        self.token = []

        print("MCTS config:")
        print(config)

        self.index = 0
        self.trying_const_range = regressor.trying_const_range
        self.trying_const_num = config["trying_const_num"] if use_const else 0
        self.trying_const_n_try = config["trying_const_n_try"]

        self.dc = config["dc"]

        self.root_node = MonteCarloNode(
            variables,
            self.operators_op,
            0,
            self.max_depth - self.trying_const_num,
            self.trying_const_range,
            self.trying_const_num,
            self.trying_const_n_try,
            self.use_const,
            self.dc,
        )

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
        tokens = self.root_node.MC()
        return "nan", tokens

    def reward(self, reward, expressions):
        last_node.backpropagate(reward, 1)
        return


class MonteCarloNode:
    """MonteCarloNode for PSRN"""

    def __init__(
        self,
        expr,
        operators_op,
        index,
        max_depth,
        trying_const_range,
        trying_const_num,
        trying_const_n_try,
        use_const,
        dc,
    ):

        self.use_const = use_const
        self.dc = dc

        self.expr = expr

        self.max_depth = max_depth

        self.n_u = 0
        self.n_b = 0

        self.operators_op = operators_op
        for op in operators_op:
            if op.is_unary:
                self.n_u += 1
            else:
                self.n_b += 1

        self.index = index

        self.trying_const_range = trying_const_range
        self.trying_const_n_try = trying_const_n_try
        self.trying_const_num = trying_const_num

        self.t = 0
        self.n = 0

        self.n_variable = len(expr)

        self.len_u_block = self.n_variable * self.n_u
        self.len_b_block = self.n_variable * (self.n_variable + 1) // 2 * self.n_b

        if self.next_is_const_child():
            self.n_new_expr = 1
        else:
            self.n_new_expr = self.len_b_block + self.len_u_block

        self.visit = False

        self.children = []
        self.father = None

        self.is_expanded = False

        self.ablation_random_MCTS = False

    def is_terminal_state(self):
        if len(self.expr) == self.max_depth + self.trying_const_num:
            return True

        else:
            return False

    def next_is_const_child(self):
        if len(self.expr) == self.max_depth:
            return True

        else:
            return False

    def is_visited(self):
        return self.visit

    def create_a_random_child(self):
        index = np.random.randint(self.n_new_expr)
        for child in self.children:
            if child.index == index:
                return False, child
        if self.next_is_const_child():
            return True, self.create_a_child_const(index)
        else:
            return True, self.create_a_child(index)

    def create_a_child(self, index):
        if index < self.len_b_block:
            factor = (self.n_variable) * (self.n_variable + 1) // 2
            op = self.operators_op[index // factor]

            v_index_1 = triu_ls[self.n_variable - 1][0][index % factor]
            v_index_2 = triu_ls[self.n_variable - 1][1][index % factor]
            expr_1 = self.expr[v_index_1]
            expr_2 = self.expr[v_index_2]
            if random.random() < 0.5:
                new_expr = op.get_expr(expr_1, expr_2)
            else:
                new_expr = op.get_expr(expr_2, expr_1)
        else:

            index -= self.len_b_block

            op = self.operators_op[self.n_b + index // self.n_variable]
            v_index = index % self.n_variable
            expr = self.expr[v_index]
            new_expr = op.get_expr(expr)

        expr_sympy = sympy.sympify(new_expr)
        for e in self.expr:
            if sympy.S(e) == expr_sympy:
                return None
        if expr_sympy == sympy.sympify("0"):
            return None
        elif expr_sympy.count_ops() > 10:
            return None
        else:
            new_node = MonteCarloNode(
                self.expr + [new_expr],
                self.operators_op,
                index,
                self.max_depth,
                self.trying_const_range,
                self.trying_const_num,
                self.trying_const_n_try,
                self.use_const,
                self.dc,
            )
        return new_node

    def create_a_child_const(self, index):
        new_const_ls = np.random.uniform(
            low=self.trying_const_range[0],
            high=self.trying_const_range[1],
            size=(self.trying_const_num,),
        ).tolist()
        n_const = len(new_const_ls)

        new_expr_ls = [str(num) for num in new_const_ls]
        new_node = MonteCarloNode(
            self.expr + new_expr_ls,
            self.operators_op,
            index,
            self.max_depth,
            self.trying_const_range,
            self.trying_const_num,
            self.trying_const_n_try,
            self.use_const,
            self.dc,
        )
        return new_node

    def backpropagate(self, t_add, n_add):
        self.t += t_add
        self.n += n_add
        self.visit = True
        if not self.father is None:
            self.father.backpropagate(t_add, n_add)
        else:
            global N
            N += n_add

    def expand(self):
        for index in range(self.n_new_expr):
            if self.next_is_const_child():
                new_node = self.create_a_child_const(index)
            else:
                new_node = self.create_a_child(index)
            if new_node is None:
                continue
            is_exist = False
            for child in self.children:
                if set(new_node.expr) == set(child.expr):
                    is_exist = True
                    break
                else:
                    pass
            if not is_exist:
                new_node.father = self
                self.children.append(new_node)
        self.is_expanded = True

    def select(self, c=2, random_select=False):
        if random_select:
            idx = np.random.randint(len(self.children))
            return self.children[idx]
        else:
            max_ucb_index = -1
            max_ucb_value = -1e20
            for i, child in enumerate(self.children):
                ucb = child.t / (child.n + 1e-6) + c * math.sqrt(
                    math.log(N) / (child.n + 1e-6)
                )
                if ucb > max_ucb_value:
                    max_ucb_value = ucb
                    max_ucb_index = i
            print("max_ucb_value", max_ucb_value)
            # print('self.children',self.children)
            return self.children[max_ucb_index]

    def MC(self):

        expr_ls = []
        if not self.is_expanded:
            self.expand()

        select_node = self.select(random_select=self.ablation_random_MCTS)
        if select_node.is_visited() and not select_node.is_terminal_state():
            select_node.expand()
            expr_ls = select_node.MC()
        else:
            current_node = select_node
            while not current_node.is_terminal_state():
                while True:
                    is_new_node, new_node = current_node.create_a_random_child()
                    if new_node != None:
                        break
                if is_new_node:
                    new_node.father = current_node
                    current_node.children.append(new_node)
                else:
                    pass
                current_node = new_node

            if self.use_const:
                new_const_ls = np.random.uniform(
                    low=self.trying_const_range[0],
                    high=self.trying_const_range[1],
                    size=(self.trying_const_num,),
                ).tolist()
                for i in range(len(new_const_ls)):
                    len_step = self.dc
                    new_const_ls[i] = round(
                        round(new_const_ls[i] / len_step) * len_step + self.dc, 2
                    )
                n_const = len(new_const_ls)

                new_expr_ls = [str(num) for num in new_const_ls]

                current_node.expr[-self.trying_const_num :] = new_expr_ls

            expr_ls = current_node.expr.copy()
            global last_node
            last_node = current_node
        assert len(expr_ls) > 0
        return expr_ls
