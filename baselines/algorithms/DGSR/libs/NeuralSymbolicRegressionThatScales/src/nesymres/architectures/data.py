import numpy as np
import sympy
import warnings
import torch
from torch.utils import data
import math
from nesymres.utils import load_metadata_hdf5, load_eq
from sympy.core.rules import Transform
from sympy import sympify, Float, Symbol
# from torch.multiprocessing import Manager
from numpy import (
    log,
    cosh,
    sinh,
    exp,
    cos,
    tanh,
    sqrt,
    sin,
    tan,
    arctan,
    nan,
    pi,
    e,
    arcsin,
    arccos,
)

import types
from typing import List
# from torch.distributions.uniform import Uniform
from ..dataset.data_utils import sample_symbolic_constants
from ..dataset.generator import Generator, UnknownSymPyOperator
# import torch.nn as nn
# import torch.nn.functional as F
import pytorch_lightning as pl
from nesymres.dclasses import DatasetDetails, Equation
from functools import partial
from ordered_set import OrderedSet
from pathlib import Path
import hydra
import traceback
from nesymres.architectures.data_utils import eq_sympy_prefix_to_token_library
import logging
from time import time
from collections import Counter

logger = logging.getLogger()


class NesymresDataset(data.Dataset):
    def __init__(
        self,
        data_path: Path,
        cfg,
        mode: str,
        return_true_eq = False
    ):
        #m = Manager()
        #self.eqs = m.dict({i:eq for i, eq in enumerate(data.eqs)})
        metadata = load_metadata_hdf5(hydra.utils.to_absolute_path(data_path))
        cfg.total_variables = metadata.total_variables
        cfg.total_coefficients = metadata.total_coefficients
        self.len = metadata.total_number_of_eqs
        self.eqs_per_hdf = metadata.eqs_per_hdf
        self.word2id = metadata.word2id
        self.data_path = data_path
        self.mode = mode
        self.cfg = cfg
        self.return_true_eq = return_true_eq
        # self.library = library

    def __getitem__(self, index):
        # print('get ', torch.utils.data.get_worker_info())
        eq = load_eq(self.data_path, index, self.eqs_per_hdf)
        code = types.FunctionType(eq.code, globals=globals(), name="f")
        consts, initial_consts = sample_symbolic_constants(
            eq, self.cfg.constants)
        if self.cfg.predict_c:
            eq_string = eq.expr.format(**consts)
        else:
            eq_string = eq.expr.format(**initial_consts)
        coeff_dict = consts
        try:
            eq_sympy_infix = constants_to_placeholder(eq_string)
            eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)
        except UnknownSymPyOperator as e:
            print(e)
            return None
            # return Equation(code=code,expr=[],coeff_dict=consts,variables=eq.variables,support=eq.support, valid=False)
        except RecursionError as e:
            print(e)
            return None
            # return Equation(code=code,expr=[],coeff_dict=consts,variables=eq.variables,support=eq.support, valid=False)
        curr_p = number_of_support_points(self.cfg.max_number_of_points, self.cfg.type_of_sampling_points)
        # if not eq.support:
        sym = []
        for sy in self.cfg.total_variables:
            if sy in eq.variables:
                curr = np.random.uniform(self.cfg.fun_support.min, self.cfg.fun_support.max, int(curr_p))
            else:
                curr = np.zeros(int(curr_p))
            sym.append(curr)
        support = np.stack(sym)
        consts_ = []
        for c in self.cfg.total_coefficients:
            if c[:2] == "cm":
                if c in coeff_dict:
                    curr = np.ones([int(curr_p)]) * coeff_dict[c]
                else:
                    curr = np.ones([int(curr_p)])
            elif c[:2] == "ca":
                if c in coeff_dict:
                    curr = np.ones([int(curr_p)]) * coeff_dict[c]
                else:
                    curr = np.zeros([int(curr_p)])
            consts_.append(curr)
        consts = np.stack(consts_)
        input_lambdi = np.float32(np.concatenate((support, consts), axis=0))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = code(*input_lambdi)
                if type(raw) == np.ndarray and raw.dtype == np.float32:
                    value = np.concatenate([support, np.expand_dims(raw,0)],0)
                    # value = np.expand_dims(np.concatenate([support, np.expand_dims(raw,0)],0),0)
                else:
                    return None
        except NameError as e:
            logger.exception('Name Error')
            print(e)
            return None
        except RuntimeError as e:
            logger.exception('Runtime Error')
            print(e)
            return None
        except Exception as e:
            logger.exception('Exception Error')
            print(e)
            return None
        if not np.sum(np.count_nonzero(np.isnan(value), axis=1), axis=0) < curr_p / 25:
            return None
        if not np.sum(np.count_nonzero(np.abs(value) > 5e4, axis=1), axis=0) < curr_p / 25:
            return None
        counts = Counter(eq_sympy_prefix)
        if counts['sin'] + counts['cos'] >= 2:
            return None
        if counts['log'] >= 2:
            return None
        if counts['exp'] >= 2:
            return None
        idx = np.repeat(np.expand_dims(np.argsort(value[-1,:]), 0), value.shape[0], axis=0)
        res = np.take_along_axis(value, idx, axis=1)
        if not np.sum(np.count_nonzero(np.isnan(res), axis=0)) == 0:
            return None
        if not np.sum(np.count_nonzero(np.abs(res) > 5e4, axis=0)) == 0:
            return None
        if not self.return_true_eq:
            return res, eq_sympy_prefix
        else:
            return res, eq_sympy_prefix, str(eq_sympy_infix)

    def __len__(self):
        return self.len

def custom_collate_fn(in_, cfg, return_true_eq = False):
    # print('collate ', torch.utils.data.get_worker_info())
    data_ = [d[0] for d in in_ if d is not None]
    if len(data_) == 0:
        return torch.Tensor([]), torch.Tensor([])
    data = torch.tensor(np.stack([d[0] for d in in_ if d is not None]), dtype=torch.float32)
    tokens = [d[1] for d in in_ if d is not None]
    if not return_true_eq:
        return data, tokens
    else:
        true_eqs = [d[2] for d in in_ if d is not None]
        return data, tokens, true_eqs
    


def constants_to_placeholder(s, symbol="c"):
    sympy_expr = sympify(s)  # self.infix_to_sympy("(" + s + ")")
    sympy_expr = sympy_expr.xreplace(
        Transform(
            lambda x: Symbol(symbol, real=True, nonzero=True),
            lambda x: isinstance(x, Float),
        )
    )
    return sympy_expr


def tokenize(prefix_expr: list, word2id: dict) -> list:
    tokenized_expr = []
    tokenized_expr.append(word2id["S"])
    for i in prefix_expr:
        tokenized_expr.append(word2id[i])
    tokenized_expr.append(word2id["F"])
    return tokenized_expr


def de_tokenize(tokenized_expr, id2word: dict):
    prefix_expr = []
    for i in tokenized_expr:
        if "F" == id2word[i]:
            break
        else:
            prefix_expr.append(id2word[i])
    return prefix_expr


def tokens_padding(tokens, padding_token):
    max_len = max([len(y) for y in tokens])
    p_tokens = torch.ones(len(tokens), max_len) * padding_token
    for i, y in enumerate(tokens):
        y = torch.tensor(y).long()
        p_tokens[i, :] = torch.cat(
            [y, torch.ones(max_len - y.shape[0]).long() * padding_token])
    return p_tokens


def number_of_support_points(p, type_of_sampling_points):
    if type_of_sampling_points == "constant":
        curr_p = p
    elif type_of_sampling_points == "logarithm":
        raise ValueError("Not supported - come back to implement")
        curr_p = int(10 ** Uniform(1, math.log10(p)).sample())
    else:
        raise NameError
    return curr_p


def sample_support(eq, curr_p, cfg):
    sym = []
    if not eq.support:
        # Seems logarithmic sampling needs to be supported
        distribution = np.random.uniform(cfg.fun_support.min, cfg.fun_support.max)
    else:
        raise NotImplementedError

    for sy in cfg.total_variables:
        if sy in eq.variables:
            curr = distribution.sample([int(curr_p)])
        else:
            curr = torch.zeros(int(curr_p))
        sym.append(curr)
    return torch.stack(sym)


def sample_constants(eq, curr_p, cfg):
    consts = []
    #eq_c = set(eq.coeff_dict.values())
    for c in cfg.total_coefficients:
        if c[:2] == "cm":
            if c in eq.coeff_dict:
                curr = torch.ones([int(curr_p)]) * eq.coeff_dict[c]
            else:
                curr = torch.ones([int(curr_p)])
        elif c[:2] == "ca":
            if c in eq.coeff_dict:
                curr = torch.ones([int(curr_p)]) * eq.coeff_dict[c]
            else:
                curr = torch.zeros([int(curr_p)])
        consts.append(curr)
    return torch.stack(consts)


def evaluate_and_wrap(eqs: List[Equation], cfg):
    if len(eqs) == 0:
        return torch.Tensor([]), torch.Tensor([])
    vals = []
    cond0 = []
    tokens_eqs = [eq.tokenized for eq in eqs]
    curr_p = number_of_support_points(
        cfg.max_number_of_points, cfg.type_of_sampling_points)
    for eq in eqs:
        support = sample_support(eq, curr_p, cfg)
        consts = sample_constants(eq, curr_p, cfg)
        input_lambdi = torch.cat([support, consts], axis=0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aaaa = eq.code(*input_lambdi)
                if type(aaaa) == torch.Tensor and aaaa.dtype == torch.float32:
                    vals.append(
                        torch.cat(
                            [support, torch.unsqueeze(aaaa, axis=0)], axis=0
                        ).unsqueeze(0)
                    )
                    cond0.append(True)
                else:
                    cond0.append(False)
        except NameError as e:
            # print(e)
            cond0.append(False)
        except RuntimeError as e:
            cond0.append(False)
        # except:
        #     breakpoint()
    tokens_eqs = [tokens_eqs[i] for i, c in enumerate(cond0) if c]
    # [equations valid out of 25 i.e. batch size, 4 (covariates + y), support points]
    if len(vals) == 0:
        return torch.Tensor([]), torch.Tensor([])
    num_tensors = torch.cat(vals, axis=0)
    # Condition for nan's consisting of less than 25% of the output y points
    cond = (
        torch.sum(torch.count_nonzero(torch.isnan(num_tensors), dim=2), dim=1)
        < curr_p / 25
    )
    num_fil_nan = num_tensors[cond]
    tokens_eqs = [tokens_eqs[i] for i, c in enumerate(cond) if c]
    # Condition for values less than 5e4 less than 25% of the output y points
    cond2 = (
        torch.sum(
            torch.count_nonzero(torch.abs(num_fil_nan) > 5e4, dim=2), dim=1
        )
        < curr_p / 25
    )
    num_fil_nan_big = num_fil_nan[cond2]
    tokens_eqs = [tokens_eqs[i] for i, c in enumerate(cond2) if c]
    idx = torch.argsort(num_fil_nan_big[:, -1, :]).unsqueeze(1).repeat(1, num_fil_nan_big.shape[1], 1)
    res = torch.gather(num_fil_nan_big, 2, idx)
    res = res[:, :, torch.sum(torch.count_nonzero(torch.isnan(res), dim=1), dim=0) == 0]
    res = res[:,:,torch.sum(torch.count_nonzero(torch.abs(res) > 5e4, dim=1), dim=0) == 0]
    return res, tokens_eqs


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_train_path,
        data_val_path,
        data_test_path,
        cfg
    ):
        super().__init__()
        self.cfg = cfg
        self.data_train_path = data_train_path
        self.data_val_path = data_val_path
        self.data_test_path = data_test_path

    def setup(self, stage=None):
        """called one ecah GPU separately - stage defines if we are at fit or test step"""
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == "fit" or stage is None:
            if self.data_train_path:
                self.training_dataset = NesymresDataset(
                    self.data_train_path,
                    self.cfg.dataset_train,
                    mode="train"
                )

            if self.data_val_path:
                self.validation_dataset = NesymresDataset(
                    self.data_val_path,
                    self.cfg.dataset_val,
                    mode="val"
                )

            if self.data_test_path:
                self.test_dataset = NesymresDataset(
                    self.data_test_path, self.cfg.dataset_test,
                    mode="test"
                )

    def train_dataloader(self):
        """returns training dataloader"""
        trainloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=partial(custom_collate_fn, cfg=self.cfg.dataset_train),
            num_workers=self.cfg.num_of_workers,
            pin_memory=True
        )
        return trainloader

    def val_dataloader(self):
        """returns validation dataloader"""
        validloader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=partial(custom_collate_fn, cfg=self.cfg.dataset_val),
            num_workers=self.cfg.num_of_workers,
            pin_memory=True,
            drop_last=False
        )
        return validloader

    def test_dataloader(self):
        """returns validation dataloader"""
        testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=partial(custom_collate_fn, cfg=self.cfg.dataset_test),
            num_workers=self.cfg.num_of_workers,
            pin_memory=True,
            drop_last=False
        )

        return testloader
