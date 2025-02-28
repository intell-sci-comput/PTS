from typing import Dict, Tuple

from ..config import Config
from ..expr_utils.exp_tree import PreTree
from ..expr_utils.calculator import cal_expression
from .utils import deap_to_tokens
from ..expr_utils.exp_tree_node import Expression
from functools import lru_cache

from numba import jit
import numpy as np
from functools import lru_cache
from deap import gp

import array

class Agent:
    """
    Genetic Algorithm Implementation Class
    """

    def __init__(self, toolbox, config_s):
        self.config_s: Config = config_s
        self.toolbox = toolbox

        self.exp_last: str = ""

        self.max_parameter = config_s.gp.max_const
        self.discount = config_s.gp.token_discount
        self.expression_dict: Dict[int, Expression] = config_s.exp_dict

        self._all_keys = set(self.expression_dict.keys())

        self.cached_fitness = lru_cache(maxsize=10000)(self._fitness)
        self.token_to_index = {token: i for i, token in enumerate(self.expression_dict.keys())}
        self.index_to_expression = [self.expression_dict[token] for token in self.expression_dict.keys()]
        self.available_cache = lru_cache(maxsize=1000)(self._available)

    @lru_cache(maxsize=1000)
    def _available(self, tree_state: Tuple[int, ...]) -> Tuple[int, ...]:
        tree = PreTree()
        for idx in tree_state:
            tree.add_exp(self.index_to_expression[idx])
        return tuple(self.token_to_index[token] for token in self.available(tree))

    def change_form(self, expr: str) -> None:
        """
        Replacing the new expression pattern
        :param expr: mode
        """
        self.exp_last = expr

    def _fitness(self, token_indices: Tuple[int, ...]) -> float:
        """
        Calculate the fitness of the individual(expression)
        """
        if len(token_indices) <= 5:
            return 1e999

        tree_state = array.array('I')
        tree = PreTree()

        for token_index in token_indices:
            if token_index not in self.available_cache(tuple(tree_state)):
                return 1e999
            tree_state.append(token_index)
            tree.add_exp(self.index_to_expression[token_index])

        symbols = tree.get_exp()
        if self.exp_last:
            symbols = f"{self.exp_last}({symbols})"

        ans = cal_expression(symbols, len(token_indices), self.config_s)
        val = self.discount ** (-len(token_indices)) * ans
        return val

    def fitness(self, individual) -> Tuple[float,]:
        """
        Wrapper function to handle caching and exceptions
        """
        try:
            token_list = deap_to_tokens(individual)
            token_indices = tuple(self.token_to_index[token] for token in token_list)
            return (self.cached_fitness(token_indices),)
        except TimeoutError:
            pass
        return (1e999,)

    @staticmethod
    def primitive_to_string(primitive):
        """
        Convert a DEAP primitive to a string representation
        """
        if isinstance(primitive, gp.Primitive):
            return primitive.name
        elif isinstance(primitive, gp.Terminal):
            return str(primitive.value)
        else:
            return str(primitive)

    @lru_cache(maxsize=None)
    def _get_excluded_nodes(self, tree_head_token, tri_count, const_num):
        exps = set()
        if tri_count > 0:
            exps.update(["Cos", "Sin"])
        if tree_head_token == "Exp":
            exps.add('Log')
        elif tree_head_token == "Log":
            exps.add('Exp')
        if const_num == self.max_parameter:
            exps.add("C")
        
        return {i for i, j in self.expression_dict.items() if j.type_name in exps}

    def available(self, tree):
        """
        Determining the usable nodes of a tree
        :param tree: expression tree
        """
        excluded = self._get_excluded_nodes(tree.head_token, tree.tri_count, tree.const_num)
        return self._all_keys - excluded
