from typing import List

import numpy as np
from deap import gp


def deap_to_tokens(individual: gp.PrimitiveTree) -> np.array:
    """
    Convert individual to tokens.
    :param individual: gp.PrimitiveTree The DEAP individual.
    :return: np.array The tokens corresponding to the individual.
    """
    tokens = np.array([i.name[3:] for i in individual], dtype=np.int32)
    return tokens


def tokens_to_deap(tokens: List[int], pset) -> gp.PrimitiveTree:
    """
    Convert list of tokens into DEAP individual.
    :param tokens: Tokens corresponding to the individual.
    :param pset: Primitive set upon which to build the individual.
    :return: The DEAP individual.
    """
    plist = [pset.mapping[f"exp{t}"] for t in tokens]
    individual = gp.PrimitiveTree(plist)
    return individual


def multi_mutate(individual: gp.PrimitiveTree, expr, pset):
    """
    Randomly select one of four types of mutation
    :param individual: DEAP individual
    :param expr:  A function object that can generate an expression when called.
    :param pset: Primitive set upon which to build the individual
    :return: DEAP individual after mutation
    """
    v = np.random.randint(0, 4)
    if v == 0:
        individual = gp.mutUniform(individual, expr, pset)
    elif v == 1:
        individual = gp.mutNodeReplacement(individual, pset)
    elif v == 2:
        individual = gp.mutInsert(individual, pset)
    elif v == 3:
        individual = gp.mutShrink(individual)

    return individual


def pre_to_level(token_list, expr_dict):
    """
    Pre-order to Level-order Traversal
    :param token_list: List of Pre-order traversal
    :param expr_dict: Expression Dictionary
    """
    son = [[] for _ in token_list]
    stack = []
    for idx, token in enumerate(token_list):
        while stack and len(son[stack[-1][0]]) == expr_dict[stack[-1][1]].child:
            stack.pop()
        if stack:
            son[stack[-1][0]].append(idx)
        stack.append((idx, token))
    ans = []
    queue = [0]
    for i in queue:
        ans.append(token_list[i])
        for s in son[i]:
            queue.append(s)
    return ans


def level_to_pre(token_list, expr_dict):
    """
    Level-order to Pre-order Traversal
    :param token_list: List of Level-order Traversal
    :param expr_dict: Expression Dictionary
    """
    son = [[] for _ in token_list]
    queue = []
    for idx, token in enumerate(token_list):
        while queue and len(son[queue[0][0]]) == expr_dict[queue[0][1]].child:
            queue.pop(0)
        if queue:
            son[queue[0][0]].append(idx)
        queue.append((idx, token))
    ans = []
    stack = [0]
    while stack:
        i = stack[-1]
        stack.pop()
        ans.append(token_list[i])
        for s in son[i]:
            stack.append(s)
    return ans
