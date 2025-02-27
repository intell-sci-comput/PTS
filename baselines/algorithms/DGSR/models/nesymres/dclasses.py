from dataclasses import dataclass
from types import CodeType
from typing import List, Tuple

import torch


@dataclass
class Equation:
    code: CodeType
    expr: str
    coeff_dict: dict
    variables: list  # FIXME
    support: tuple = None  # pyright: ignore
    tokenized: list = None  # pyright: ignore
    valid: bool = True
    number_of_points: int = None  # pyright: ignore


@dataclass
class NNEquation:
    numerical_values: torch.tensor  # pyright: ignore
    tokenized: torch.tensor  # pyright: ignore
    expr: List[str]


@dataclass
class DataModuleParams:
    max_number_of_points: int
    type_of_sampling_points: str
    support_extremes: Tuple
    constant_degree_of_freedom: int
    predict_c: bool
    distribution_support: str
    input_normalization: bool


# @dataclass
# class Dataset:
#     eqs: List[Equation]
#     config: dict
#     total_coefficients: list
#     total_variables: list
#     word2id: dict
#     id2word: dict
#     una_ops: list
#     bin_ops: list
#     rewrite_functions: list
#     unique_index: set = None
#     total_number_of_eqs: int = 0


@dataclass
class GeneratorDetails:
    max_len: int
    operators: str
    max_ops: int
    # int_base: int
    # precision: int
    rewrite_functions: str
    variables: list
    eos_index: int
    pad_index: int


@dataclass
class DatasetDetails:
    # eqs: List[Equation]
    config: dict
    total_coefficients: list
    total_variables: list
    word2id: dict
    id2word: dict
    una_ops: list
    bin_ops: list
    rewrite_functions: list
    total_number_of_eqs: int
    eqs_per_hdf: int
    generator_details: GeneratorDetails
    unique_index: set = None  # pyright: ignore


@dataclass
class BFGSParams:
    activated: bool = True
    n_restarts: bool = 10  # pyright: ignore
    add_coefficients_if_not_existing: bool = False
    normalization_o: bool = False
    idx_remove: bool = True
    normalization_type: str = ["MSE", "NMSE"][0]
    stop_time: int = 1e9  # pyright: ignore


@dataclass
class FitParams:
    word2id: dict
    id2word: dict
    total_coefficients: list
    total_variables: list
    rewrite_functions: list
    una_ops: list = None  # pyright: ignore
    bin_ops: list = None  # pyright: ignore
    bfgs: BFGSParams = BFGSParams()
    beam_size: int = 2


# @dataclass
# class ConstantsOptions:
#     max_constants: int
#     min_additive_constant_support: float
#     max_additive_constant_support: float
#     min_multiplicative_constant_support: float
#     max_multiplicative_constant_support: float
