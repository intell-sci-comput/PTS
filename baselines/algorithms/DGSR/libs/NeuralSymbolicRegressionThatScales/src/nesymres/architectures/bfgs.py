import os
import numpy as np
import random
import math
from scipy.optimize import minimize
import types
import click
import marshal
import copyreg
import sys
import ast
import pdb
from torch.utils.data import DataLoader, random_split
import torch
from torch import nn
import torch.nn.functional as F
import sympy as sp
from dataclasses import dataclass
from ..dataset.generator import Generator
from . import data
from typing import Tuple
import time
import re
from ..dataset.sympy_utils import add_multiplicative_constants, add_additive_constants



class TimedFun:
    def __init__(self, fun, stop_after=10):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after

    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            raise ValueError("Time is over.")
        self.fun_value = self.fun_in(*x, *args)
        self.x = x
        return self.fun_value


def bfgs(pred_str, X, y, cfg):

    #Check where dimensions not use, and replace them with 1 to avoid numerical issues with BFGS (i.e. absent variables placed in the denominator)
    y = y.squeeze()
    X = X.clone()
    bool_dim = (X==0).all(axis=1).squeeze()
    X[:,:,bool_dim] = 1 

    pred_str = pred_str[1:].tolist()
    #pred_str = [x if x<14 else x+1 for x in pred_str]
    raw = data.de_tokenize(pred_str, cfg.id2word)
    # if "constant" in prefix:
    #     for j,i in enumerate(list(pred_str)[:-1]):
    #         if i == "constant":    
    #             expre[j] = 'c{}'.format(str(c))
    #             c=c+1
    #     example = "".join(list(expre))  

    if cfg.bfgs.add_coefficients_if_not_existing and 'constant' not in raw:           
        print("No constants in predicted expression. Attaching them everywhere")
        variables = {x:sp.Symbol(x, real=True, nonzero=True) for x in cfg.total_variables}
        infix = Generator.prefix_to_infix(raw, coefficients=cfg.total_coefficients, variables=cfg.total_variables)
        s = Generator.infix_to_sympy(infix,variables, cfg.rewrite_functions)
        placeholder = {x:sp.Symbol(x, real=True,nonzero=True) for x in ["cm","ca"]}
        s = add_multiplicative_constants(s, placeholder["cm"], unary_operators=cfg.una_ops)
        s = add_additive_constants(s,  placeholder, unary_operators=cfg.una_ops)
        s = s.subs(placeholder["cm"],0.43)
        s = s.subs(placeholder["ca"],0.421)
        s_simplified = data.constants_to_placeholder(s,symbol="constant")
        prefix = Generator.sympy_to_prefix(s_simplified)
        candidate = Generator.prefix_to_infix(prefix, 
                                        coefficients=["constant"], 
                                        variables=cfg.total_variables)
    else:
        candidate = Generator.prefix_to_infix(raw, 
                                        coefficients=["constant"], 
                                        variables=cfg.total_variables)
    candidate = candidate.format(constant="constant")
    
    c = 0 
    expr = candidate
    for i in range(candidate.count("constant")):
        expr = expr.replace("constant", f"c{i}",1)



    
    print('Constructing BFGS loss...')

    if cfg.bfgs.idx_remove:
        print('Flag idx remove ON, Removing indeces with high values...')
        bool_con = (X<200).all(axis=2).squeeze() 
        X = X[:,bool_con,:]
        # idx_leave = np.where((np.abs(input_batch[:,3].numpy()))<200)[0]
        # xx = xx[:,idx_leave]
        # input_batch = input_batch[idx_leave,:]


    max_y = np.max(np.abs(torch.abs(y).cpu().numpy()))
    print('checking input values range...')
    if max_y > 300:
        print('Attention, input values are very large. Optimization may fail due to numerical issues')

    diffs = []
    for i in range(X.shape[1]):
        curr_expr = expr
        for idx, j in enumerate(cfg.total_variables):
            curr_expr = sp.sympify(curr_expr).subs(j,X[:,i,idx]) 
        diff = curr_expr - y[i]
        diffs.append(diff)
    #         breakpoint()
    # diff = [sp.sympify(example).replace(y,xx[1,i]).replace(x,xx[0,i]).replace(z,xx[2,i])-input_batch[i,-1] for i in range(input_batch.shape[0])]
    if cfg.bfgs.normalization_o:
        raise NotImplementedError
        diff = [x/max_eq for x in diffs]
        #diff = [sympify(example).replace(y,xx[1,i]).replace(x,xx[0,i]).replace(z,xx[2,i])-input_batch[i,-1]/max_eq for i in range(input_batch.shape[0])]
    loss = 0
    cnt = 0
    if cfg.bfgs.normalization_type == "NMSE": # and (mean != 0):
        mean_y = np.mean(y.numpy())
        if abs(mean_y) < 1e-06:
            print("Normalizing by a small value")
        loss = (np.mean(np.square(diffs)))/mean_y  ###3 check
    elif cfg.bfgs.normalization_type == "MSE": 
        loss = (np.mean(np.square(diffs)))
    else:
        raise KeyError
    
    print('Loss constructed, starting new BFGS optmization...') 

    # Lists where all restarted will be appended
    F_loss = []
    consts_ = []
    funcs = []
    symbols = {i: sp.Symbol(f'c{i}') for i in range(candidate.count("constant"))}
    
    for i in range(cfg.bfgs.n_restarts):
        # Compute number of coefficients
        x0 = np.random.randn(len(symbols))
        s = list(symbols.values())
        #bfgs optimization
        fun_timed = TimedFun(fun=sp.lambdify(s,loss, modules=['numpy']), stop_after=cfg.bfgs.stop_time)
        if len(x0):
            minimize(fun_timed.fun,x0, method='BFGS')   #check consts interval and if they are int
            consts_.append(fun_timed.x)
        else:
            consts_.append([])
        final = expr
        for i in range(len(s)):
            final = sp.sympify(final).replace(s[i],fun_timed.x[i])
        if cfg.bfgs.normalization_o:
            funcs.append(max_y*final)
        else:
            funcs.append(final)
        
        values = {x:X[:,:,idx].cpu() for idx, x in enumerate(cfg.total_variables)} #CHECK ME
        y_found = sp.lambdify(",".join(cfg.total_variables), final)(**values)
        final_loss = np.mean(np.square(y_found-y.cpu()).numpy())
        F_loss.append(final_loss)

    try:
        k_best = np.nanargmin(F_loss)
    except ValueError:
        print("All-Nan slice encountered")
        k_best = 0
    return funcs[k_best], consts_[k_best], F_loss[k_best], expr
            