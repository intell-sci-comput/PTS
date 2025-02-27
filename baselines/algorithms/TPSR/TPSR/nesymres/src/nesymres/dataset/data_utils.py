from typing import Tuple
import torch
from torch._C import Value
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
import numpy as np
import random
import warnings
import inspect
from torch.distributions.uniform import Uniform
import math
import types
from numpy import log, cosh, sinh, exp, cos, tanh, sqrt, sin, tan, arctan, nan, pi, e, arcsin, arccos
from sympy import sympify,lambdify, Symbol
from sympy import Float
from ..dclasses import Equation


def create_uniform_support(sampling_distribution, n_variables, p):
    sym = {}
    for idx in range(n_variables):
        sym[idx] = sampling_distribution.sample([int(p)])
    support = torch.stack([x for x in sym.values()])
    return support


def group_symbolically_indetical_eqs(data,indexes_dict,disjoint_sets):
    for i, val in enumerate(data.eqs):
        if not val.expr in indexes_dict:
            indexes_dict[val.expr].append(i)
            disjoint_sets[i].append(i)
        else:
            first_key = indexes_dict[val.expr][0]
            disjoint_sets[first_key].append(i)
    return indexes_dict, disjoint_sets


def dataset_loader(train_dataset, test_dataset, batch_size=1024, valid_size=0.20):
    num_train = len(train_dataset)
    num_test_h = len(test_dataset)
    indices = list(range(num_train))
    test_idx_h = list(range(num_test_h))
    np.random.shuffle(test_idx_h)
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0
    )
    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0
    )
    test_loader_h = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, valid_loader, test_loader_h, valid_idx, train_idx


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def initialize_weights(m):
    """Used for the transformer"""
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def evaluate_fun(args):
    """Single input algorithm as this function is used in multiprocessing"""
    fun ,support = args
    if type(fun)==list and not len(fun):
        return []
    f = types.FunctionType(fun, globals=globals(), name='f')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            evaled = f(*support)
            if type(evaled) == torch.Tensor and evaled.dtype == torch.float32:
                return evaled.numpy().astype('float16')
            else:
                return []
    except NameError as e:
        print(e)
        return []
    except RuntimeError as e:
        print(e)
        return []

def return_dict_metadata_dummy_constant(metadata):
    dict = {key:0 for key in metadata.total_coefficients}
    for key in dict.keys():
        if key[:2] == "cm":
            dict[key] = 1
        elif key[:2] == "ca":
            dict[key] = 0
        else:
            raise KeyError
    return dict
    



def sample_symbolic_constants(eq: Equation, cfg=None) -> Tuple:
    """Given an equation, returns randomly sampled constants and dummy contants

    Args:
      eq: an Equation.
      cfg: Used for specifying how many and in which range to sample constants. If None, consts equal to dummy_consts

    Returns:
      consts: 
      dummy_consts: 
    """

    dummy_consts = {const: 1 if const[:2] == "cm" else 0 for const in eq.coeff_dict.keys()}
    consts = dummy_consts.copy()
    if cfg:
        used_consts = random.randint(0, min(len(eq.coeff_dict),cfg.num_constants))
        symbols_used = random.sample(set(eq.coeff_dict.keys()), used_consts)
        for si in symbols_used:
            if si[:2] == "ca":
                consts[si] = round(float(Uniform(cfg.additive.min, cfg.additive.max).sample()),3)
            elif si[:2] == "cm":
                consts[si] = round(float(Uniform(cfg.multiplicative.min, cfg.multiplicative.max).sample()),3)
            else:
                raise KeyError
    else:
        consts = dummy_consts
    return consts, dummy_consts