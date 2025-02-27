"""Numerical derivatives for observed time series data
 five-point approximation

Created on Wed July 7 14:40:18 2021
@authors: Tingting Gao, Gang Yan

"""
import pandas as pd
import numpy as np 

def NumericalDeriv(TimeSeries,dim,Nnodes,deltT):
    x_center = TimeSeries[2:-2,:]
    x_PlusTwo = TimeSeries[4:,:]
    x_PlusOne = TimeSeries[3:-1,:]
    x_MinusTwo = TimeSeries[:-4,:]
    x_MinusOne = TimeSeries[1:-3,:]
    dxdt = (x_MinusTwo - 8 * x_MinusOne + 8 * x_PlusOne - x_PlusTwo) / (12 * deltT)
    T_len = len(dxdt[:,0])
    NumDiv = np.zeros(shape=(T_len*Nnodes,dim))
    for j in range(0,dim):
        for i in range(0,Nnodes):
            NumDiv[i*T_len:(i+1)*T_len,j]  = dxdt[:,dim*i+j]
    if dim == 1:
        column_values = ['dx1']
    if dim == 2:
        column_values = ['dx1','dx2']
    if dim == 3:
        column_values = ['dx1','dx2','dx3']
    if dim == 4:
        column_values = ['dx1','dx2','dx3','dx4']
    NumDiv= pd.DataFrame(data = NumDiv, columns = column_values)
    return NumDiv

def preprocess(FuncMatrix,NumDiv,dim):
    preprocess = pd.concat([FuncMatrix,NumDiv],axis = 1)
    preprocess = preprocess.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    FuncMatrix = preprocess.iloc[:,0:FuncMatrix.shape[1]]
    NumDiv = preprocess.iloc[:,-dim:]
    return FuncMatrix,NumDiv

