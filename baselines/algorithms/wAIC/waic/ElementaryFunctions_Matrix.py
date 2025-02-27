# -*- coding: UTF-8 -*-
"""
Created on Wed July 9 01:13:52 2021
@authors: Tingting Gao, Gang Yan

"""

import pandas as pd
import numpy as np
import sys
sys.path.append("../code/utils")
from ElementaryFunctionsPool import *

def ElementaryFunctions_Matrix(TimeSeries, dim, Nnodes, A, selfPolyOrder, coupledPolyOrder = 1, PolynomialIndex = True, TrigonometricIndex = True, \
    ExponentialIndex = True, FractionalIndex = True, ActivationIndex = True, RescalingIndex = True, CoupledPolynomialIndex = True, \
        CoupledTrigonometricIndex = True, CoupledExponentialIndex = True, CoupledFractionalIndex = True, \
            CoupledActivationIndex = True, CoupledRescalingIndex = True):
    
    ElementaryMatrix = pd.DataFrame()
    if PolynomialIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix,Polynomial_functions(TimeSeries, dim, Nnodes, selfPolyOrder)],axis=1)
    if TrigonometricIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Trigonometric(TimeSeries, dim, Nnodes, Sin = True, Cos = True, Tan = True)],axis=1)
    if ExponentialIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Exponential(TimeSeries, dim, Nnodes, expomential = True)],axis=1)
    if FractionalIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Fractional(TimeSeries, dim, Nnodes, fractional = True)],axis=1)
    if ActivationIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Activation(TimeSeries, dim, Nnodes, Sigmoid = True, Tanh = True, Regulation = True)],axis=1)
    if RescalingIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, rescaling(TimeSeries, dim, Nnodes, A, Rescal = True)],axis=1)
    if CoupledPolynomialIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, coupled_Polynomial_functions(TimeSeries, dim, Nnodes, A, coupledPolyOrder)],axis=1)
    if CoupledTrigonometricIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Coupled_Trigonometric_functions(TimeSeries, dim, Nnodes, A, Sine = True, Cos = False, Tan = False)],axis=1)
    if CoupledExponentialIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Coupled_Exponential_functions(TimeSeries, dim, Nnodes, A, Exponential = True)],axis=1)
    if CoupledFractionalIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Coupled_Fractional_functions(TimeSeries, dim, Nnodes, A, Fractional = True)],axis=1)
    if CoupledActivationIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Coupled_Activation_functions(TimeSeries, dim, Nnodes, A, Sigmoid = True, Tanh = True, Regulation = True)],axis=1)
    if CoupledRescalingIndex == True:
        ElementaryMatrix = pd.concat([ElementaryMatrix, Coupled_Rescaling_functions(TimeSeries, dim, Nnodes, A, Rescaling = True)],axis=1)
        
    return ElementaryMatrix
